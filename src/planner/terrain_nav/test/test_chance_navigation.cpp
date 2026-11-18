// Real planner interfaces on controlled maps; no simulator or plant.
#define TERRAIN_NAV_BENCHMARK
#include "../src/terrain_nav_node.cpp"
#include <gtest/gtest.h>
#include <rcl/time.h>
#include <future>

struct TerrainNavBenchmark
{
  static void candidate_proof_can_decline_extra_sampling_without_declining_its_upper_bound()
  {
    if(!terrain_nav::jointMotionGpuAvailable())GTEST_SKIP()<<"CUDA backend unavailable";
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cuda";
    n->motion_joint_risk_limit_=.05;
    n->motion_joint_samples_=1024;n->motion_joint_checks_=1000000;
    n->live_obstacle_inflation_radius_=0.;
    const auto body=n->collisionEnvelope(n->footprint_margin_);
    const double time=100.;auto actor=owned_actor(time);actor.radius_bound=.2;
    actor.center_x=body.front+.2+.045+.40;actor.center_y=0.;actor.vx=actor.vy=0.;
    actor.cells={{{actor.center_x,0.}}};actor.gaussian.mean<<actor.center_x,0.,0.,0.;
    actor.gaussian.covariance.setZero();actor.gaussian.covariance(0,0)=.04;
    LiveObstacleSnapshot live;live.stamp_sec=time;live.tracks={actor};
    const std::vector<terrain_nav::YieldState> path{{0.,0.,0.,0.,0.,0.},{.02,0.,0.,0.,0.,0.}};
    const auto bounded=n->jointMovingRisk(path,time,live,false,0.,false);
    ASSERT_TRUE(bounded.valid);EXPECT_EQ(bounded.samples,1024u);
    EXPECT_EQ(n->motion_joint_checks_,1000001u);
    // Exercise the refinement branch without relying on a random hit count
    // falling on one side of a hard-coded fixture threshold. This local
    // threshold is not a navigation/physical-qualification configuration.
    ASSERT_GT(bounded.upper,bounded.estimate);
    n->motion_joint_risk_limit_=(bounded.upper+bounded.estimate)*.5;
    EXPECT_GT(bounded.upper,n->motion_joint_risk_limit_);
    EXPECT_LT(bounded.estimate,n->motion_joint_risk_limit_);
    // The same first stream retains its failing UCB. Optional work obtains
    // a NEW proof and confidence allocation; it never changes that first UCB.
    n->motion_joint_checks_=1000000;
    const auto refined=n->jointMovingRisk(path,time,live);
    ASSERT_TRUE(refined.valid);EXPECT_EQ(refined.samples,65536u);
    EXPECT_EQ(n->motion_joint_checks_,1000002u);EXPECT_GT(refined.upper,0.);
    EXPECT_GT(bounded.upper,n->motion_joint_risk_limit_);
  }

  static void safe_braking_does_not_remove_an_earlier_certifiable_acceleration_window(bool unrelated_cue=false)
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->dynamic_acceleration_escape_enable_=true;n->max_speed_=1.33;n->reverse_max_speed_=.8;
    n->reverse_recovery_enable_=n->reverse_enable_=true;
    n->dynamic_time_indexed_routing_=n->unclassified_encounter_model_release_=true;
    n->unclassified_encounter_speed_=.6;n->max_steer_=.31432913231137;
    n->wheel_base_=.43;n->rear_axle_offset_=.215;
    n->dynamic_speed_dead_time_=0.;n->dynamic_steering_dead_time_=.06;
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=.6;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
    n->have_path_=true;n->path_={{0.,0.,0.,0.},{3.,0.,0.,3.}};
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=100.;
    live->min_x=live->min_y=-4.;live->width=live->height=80;live->resolution=.1;live->inflation_radius=0.;
    live->hard.assign(6400,0);live->present_hard=live->hard;
    auto actor=owned_actor(100.);actor.center_x=1.5;actor.center_y=2.7;actor.vx=0.;actor.vy=-.9;
    actor.cells={{{1.5,2.7}}};actor.radius_bound=.2;
    actor.gaussian.mean<<1.5,2.7,0.,-.9;actor.gaussian.covariance=Eigen::Matrix4d::Identity()*1e-12;
    actor.gaussian.process_density=0.;live->tracks={actor};
    live->compact_observed={{{1.475,2.675,1.525,2.725}}};
    if(unrelated_cue)live->compact_observed.push_back({2.8,2.4,3.2,2.8});
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto model=n->configuredYieldModel();terrain_nav::YieldState root;root.speed=.6;
    const auto stop=n->issuedYieldRollout(root,{{0,0,0}},model,4.5,100.);
    ASSERT_FALSE(n->earlierStoppingUnsafe(stop,100.,*live));
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0.,0.,0.,*live),unrelated_cue?.6:1.33);
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_DOUBLE_EQ(n->dynamic_proposal_speed_limit_,unrelated_cue?.6:1.33);
    if(unrelated_cue) {
      EXPECT_LE(n->last_cmd_v_,.6);
      return;
    }
    ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
    EXPECT_GT(n->last_cmd_v_,.6);
    const auto path=terrain_nav::finiteJointRollout(n->issuedYieldRollout(root,n->dynamic_commands_,model,6.,100.));
    ASSERT_FALSE(path.empty());EXPECT_TRUE(n->timedYieldSafe(path,100.,n->dynamic_commands_,*map,*live));
    // A complete-model action still cannot manufacture missing terrain.
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(path,100.,n->dynamic_commands_,*map,*live));
  }

  static void reverse_momentum_does_not_disable_a_certifiable_forward_escape()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->dynamic_acceleration_escape_enable_=true;n->max_speed_=1.33;n->reverse_max_speed_=.8;
    n->reverse_recovery_enable_=n->reverse_enable_=true;
    n->max_steer_=.31432913231137;
    n->wheel_base_=.43;n->rear_axle_offset_=.215;
    n->dynamic_speed_dead_time_=0.;n->dynamic_steering_dead_time_=.06;
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=-.4;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=100.;
    live->min_x=live->min_y=-4.;live->width=live->height=80;live->resolution=.1;live->inflation_radius=0.;
    live->hard.assign(6400,0);live->present_hard=live->hard;
    auto actor=owned_actor(100.);actor.center_x=-2.;actor.center_y=0.;actor.vx=1.;actor.vy=0.;
    actor.cells={{{-2.,0.}}};actor.radius_bound=.2;
    actor.gaussian.mean<<-2.,0.,1.,0.;actor.gaussian.covariance=Eigen::Matrix4d::Identity()*1e-12;
    actor.gaussian.process_density=0.;live->tracks={actor};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto model=n->configuredYieldModel();terrain_nav::YieldState state;state.speed=-.4;
    const std::vector<terrain_nav::YieldCommand> stop{{0,0,0}};
    const auto braking=n->issuedYieldRollout(state,stop,model,3.,100.);
    EXPECT_FALSE(n->timedYieldSafe(braking,100.,stop,*map,*live));
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_GT(n->dynamic_proposals_generated_,15u);
    ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
    EXPECT_GT(n->last_cmd_v_,0.);EXPECT_LT(n->dynamic_initial_.speed,0.);
    const auto path=terrain_nav::finiteJointRollout(n->issuedYieldRollout(state,n->dynamic_commands_,model,6.,100.));
    ASSERT_FALSE(path.empty());EXPECT_LT(path[1].x,0.);EXPECT_GT(path.back().x,.5);
    EXPECT_TRUE(n->timedYieldSafe(path,100.,n->dynamic_commands_,*map,*live));
  }

  static void unconfirmed_motion_cannot_disappear_from_a_multisecond_action_proof()
  {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";
    n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=rclcpp::Time(100000000000LL,RCL_ROS_TIME);
    LiveObstacleSnapshot live;live.stamp_sec=100.;live.min_x=live.min_y=-4.;
    live.width=live.height=80;live.resolution=.1;live.inflation_radius=0.;
    live.hard.assign(6400,0);live.present_hard=live.hard;
    terrain_nav::YieldState root;terrain_nav::YieldModel model;model.delay=0.;
    const std::vector<terrain_nav::YieldCommand> commands{{0,.4,0},{1.,0,0}};
    const auto path=terrain_nav::finiteJointRollout(terrain_nav::rolloutQueuedYield(root,commands,model,6.,100.,{}));
    ASSERT_FALSE(path.empty());ASSERT_TRUE(n->timedYieldSafe(path,100.,commands,*map,live));
    terrain_nav::MotionTrackSnapshot weak;weak.stamp=100.;weak.resolution=.05;
    weak.cells={{{.5,2.}}};weak.vy=-.8;weak.acceleration_bound=0.;
    live.provisional_tracks={weak};
    const auto short_path=terrain_nav::rolloutQueuedYield(root,commands,model,.2,100.,{});
    EXPECT_FALSE(n->unconfirmedMotionUnsafe(short_path,100.,live));
    std::string reason;
    EXPECT_FALSE(n->timedYieldSafe(path,100.,commands,*map,live,nullptr,nullptr,&reason));
    EXPECT_EQ(reason,"unconfirmed motion reachability");
    EXPECT_TRUE(live.tracks.empty());EXPECT_EQ(live.provisional_tracks.front().id,0u);
    live.provisional_tracks.front().cells={{{20.,20.}}};
    EXPECT_TRUE(n->timedYieldSafe(path,100.,commands,*map,live));
    live.provisional_tracks={weak};live.provisional_tracks.front().stamp=99.9;
    n->dynamic_observed_only_=true;
    EXPECT_TRUE(n->timedYieldSafe(path,100.,commands,*map,live));
    // Dropping a blind-zone motion cue never removes actual measured cells.
    live.present_hard[40*80+40]=1;live.hard=live.present_hard;
    EXPECT_FALSE(n->timedYieldSafe(path,100.,commands,*map,live));
  }

  static void conditional_conflict_retirement_needs_positive_identity_but_not_unrelated_missing_objects()
  {
    for(double acceleration:{0.,1.2}) {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->live_obstacle_inflation_radius_=0.;
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    OdomState odom;odom.valid=true;odom.stamp=n->now();
    auto far=owned_actor(100.);far.id=8;far.center_x=3.5;far.center_y=-2.7;far.vy=.7;
    far.cells={{{3.5,-2.7}}};far.gaussian.mean<<3.5,-2.7,0.,.7;
    far.gaussian.covariance=Eigen::Matrix4d::Identity()*.000324;far.gaussian.process_density=.04;far.acceleration_bound=1.2;
    far.gaussian.longitudinal_acceleration_bound=acceleration;
    auto near=owned_actor(100.);near.id=7;near.center_x=0.;near.center_y=1.5;near.vy=-1.;
    near.cells={{{0.,1.5}}};near.gaussian.mean<<0.,1.5,0.,-1.;
    LiveObstacleSnapshot live;live.stamp_sec=100.;live.tracks={near,far};
    n->dynamic_unresolved_={{7,100.},{8,100.},{99,100.}};
    n->dynamic_active_=true;n->dynamic_commands_={{0,.4,0},{1,0,0}};
    EXPECT_TRUE(n->retireGeometricallySeparatedConflicts(odom,100.,live));
    EXPECT_EQ(n->dynamic_unresolved_.count(8),0u);EXPECT_EQ(n->dynamic_unresolved_.count(7),1u);
    EXPECT_EQ(n->dynamic_unresolved_.count(99),1u); // Missing is NEVER cleared by this change.
    EXPECT_TRUE(n->dynamic_active_);EXPECT_EQ(n->dynamic_commands_.size(),2u);
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(odom,100.,live));
    n->dynamic_unresolved_[8]=100.;live.tracks.push_back(far);
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(odom,100.,live)); // Ambiguous source.
    live.tracks.pop_back();live.tracks[1].stamp=99.9;live.tracks[1].gaussian.stamp=99.9;
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(odom,100.,live)); // Coast is not a new measurement.
    live.tracks[1]=far;live.tracks[1].gaussian.valid=false;
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(odom,100.,live));
    }
  }
  static void pending_published_input_is_included_in_body_and_stop_admission()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    LiveObstacleSnapshot live;live.stamp_sec=100.;live.min_x=live.min_y=-4.;
    live.width=live.height=160;live.resolution=.05;live.hard.assign(25600,0);live.present_hard=live.hard;
    live.inflation_radius=0.;live.forecast.assign(live.hard.size(),0);
    // Exact raw rectangle just beyond the unchanged protected front.
    const double front=n->collisionEnvelope(n->footprint_margin_).front;
    const int ix=int((front+.025-live.min_x)/live.resolution),iy=80;
    live.present_hard[iy*160+ix]=1;live.hard=live.present_hard;
    live.rectangles.resize(live.hard.size());
    live.rectangles[iy*160+ix]={front+.025,-.02,front+.035,.02};
    terrain_nav::YieldState root;terrain_nav::YieldModel model;model.delay=.2;model.step=.005;
    const std::vector<terrain_nav::YieldCommand> stop{{0,0,0}};
    auto trajectory=n->issuedYieldRollout(root,stop,model,3.,100.);
    ASSERT_TRUE(n->timedYieldSafe(trajectory,100.,stop,*map,live)); // Measured rest alone misses it.
    ASSERT_TRUE(n->issued_commands_.record({99.,0.,0.}));
    ASSERT_TRUE(n->issued_commands_.record({99.81,1.,0.}));
    trajectory=n->issuedYieldRollout(root,stop,model,3.,100.);
    ASSERT_GT(trajectory.back().x,.04);
    EXPECT_FALSE(n->timedYieldSafe(trajectory,100.,stop,*map,live));
    live.hard.assign(25600,0);live.present_hard=live.hard;
    EXPECT_TRUE(n->timedYieldSafe(trajectory,100.,stop,*map,live));
    for(auto&slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(trajectory,100.,stop,*map,live)); // Queue cannot create ground support.
    n->steer_output_sign_=-1.;geometry_msgs::msg::Twist command;command.linear.x=.4;command.angular.z=-.2;
    n->publishCommand(command);auto issued=n->issuedCommandSnapshot();
    ASSERT_FALSE(issued.empty());EXPECT_DOUBLE_EQ(issued.back().speed,.4);EXPECT_DOUBLE_EQ(issued.back().steering,.2);
    n->localization_fault_=true;n->publishCommand(command);issued=n->issuedCommandSnapshot();
    EXPECT_DOUBLE_EQ(issued.back().speed,0.);EXPECT_DOUBLE_EQ(issued.back().steering,0.);
  }
  static terrain_nav::MotionTrackSnapshot owned_actor(double stamp)
  {
    terrain_nav::MotionTrackSnapshot t;t.id=71;t.motion_confirmed=t.circular=t.center_motion_fit=true;
    t.stamp=stamp;t.resolution=.05;t.center_x=1.025;t.center_y=.025;t.radius_bound=.12;
    t.vy=1.;t.cells={{{1.025,.025}}};t.gaussian.valid=true;t.gaussian.stamp=stamp;
    t.gaussian.observation_count=10;t.gaussian.mean<<1.025,.025,0.,1.;
    t.gaussian.covariance.setZero();t.gaussian.process_density=0.;return t;
  }
  static void moving_ownership_is_unique_measured_and_source_exact()
  {
    const auto t=owned_actor(100.);std::vector<uint8_t> raw(1600,0);raw[20*40+30]=1;
    auto labels=terrain_nav::measuredMovingCells({t},raw,40,40,-.5,-1.,.05,100.);
    ASSERT_EQ(labels.size(),1u);EXPECT_EQ(labels.at(71),(std::vector<size_t>{830}));
    auto bad=t;bad.id=72;
    EXPECT_TRUE(terrain_nav::measuredMovingCells({t,bad},raw,40,40,-.5,-1.,.05,100.).empty());
    for(int mode=0;mode<7;++mode) {
      bad=t;
      if(mode==0)bad.motion_confirmed=false;
      if(mode==1)bad.stamp-=.1;
      if(mode==2)bad.gaussian.observation_count=2;
      if(mode==3)bad.gaussian.mean.tail<2>().setZero();
      if(mode==4)bad.radius_bound=.001;
      if(mode==5)bad.cells={{{1e300,1e300}}};
      if(mode==6)bad.gaussian.valid=false;
      const auto actual=terrain_nav::measuredMovingCells({bad},raw,40,40,-.5,-1.,.05,100.);
      if(mode==3)EXPECT_EQ(actual,labels); // Confirmed identity measured while slowing remains modeled.
      else EXPECT_TRUE(actual.empty())<<mode;
    }
    bad=t;bad.motion_confirmed=false;bad.id=0;
    EXPECT_TRUE(terrain_nav::measuredMovingCells({t,bad},raw,40,40,-.5,-1.,.05,100.).empty());
    EXPECT_TRUE(terrain_nav::measuredMovingCells({t,t},raw,40,40,-.5,-1.,.05,100.).empty());
    EXPECT_FALSE(terrain_nav::rectangleInsideMeasuredCell({1.,0.,1.06,.05},{1.,0.,1.05,.05}));
  }
  static void moving_occupancy_uses_time_without_erasing_walls_unknown_or_actual_root()
  {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";
    n->dynamic_time_indexed_routing_=true;n->motion_joint_risk_limit_=.05;
    n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    map->grid.add("collision_subcell_contract",false,3.);
    map->grid.add("collision_subcell_factor",false,4.);
    map->grid.add("collision_subcell_bits",false,65535.);
    map->grid.add("terrain_failure_probability_upper",true,1e-8F);
    LiveObstacleSnapshot live;live.stamp_sec=map->stamp.seconds();live.min_x=live.min_y=-4.;
    live.width=live.height=160;live.resolution=.05;live.inflation_radius=0.;
    live.hard.assign(25600,0);const size_t actor_cell=80*160+100;live.hard[actor_cell]=1;
    live.current_measured=live.present_hard=live.uninflated_present=live.hard;live.forecast.assign(25600,0);
    live.tracks={owned_actor(live.stamp_sec)};live.moving_cells[71]={actor_cell};live.held_moving_cells=live.moving_cells;
    Eigen::Array3i index;ASSERT_TRUE(map->grid.pos2Index(Eigen::Vector3d(1.025,.025,0.),index));
    Eigen::Vector3d center;ASSERT_TRUE(map->grid.index2Pos(index,center));
    const int sx=int(std::floor((1.025-center.x()+.1)/.05)),sy=int(std::floor((.025-center.y()+.1)/.05));
    map->grid.at("collision_risk",index)=1.;map->grid.at("collision_subcell_bits",index)=1<<(sy*4+sx);
    n->labelMovingSubcells(*map,live);ASSERT_EQ(map->moving_subcells.size(),1u);
    terrain_nav::YieldState root;terrain_nav::YieldModel model;model.wheelbase=n->wheel_base_;
    model.rear_axle_offset=n->rear_axle_offset_;
    const std::vector<terrain_nav::YieldCommand> commands{{0.,.6,0.},{3.1,0.,0.}};
    const auto states=terrain_nav::rolloutYield(root,root,commands,model,5.);
    std::string failure;YieldFootprintCache cache;
    ASSERT_TRUE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,live,nullptr,nullptr,&failure,nullptr,&cache))<<failure;
    EXPECT_EQ(live.hard[actor_cell],1);EXPECT_EQ(map->grid.at("collision_risk",index),1.);
    auto trail=live;const size_t old_cell=80*160+80;
    trail.hard[old_cell]=trail.present_hard[old_cell]=trail.uninflated_present[old_cell]=1;
    trail.held_moving_cells[71].push_back(old_cell);
    EXPECT_TRUE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,trail));
    EXPECT_EQ(trail.hard[old_cell],1); // Original evidence is not mutated.
    auto ambiguous_trail=trail;ambiguous_trail.held_moving_cells[71].pop_back();
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,ambiguous_trail));
    auto actually_present=trail;actually_present.current_measured[old_cell]=1;
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,actually_present));
    auto no_raw_provenance=trail;no_raw_provenance.current_measured.clear();
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,no_raw_provenance));
    // No current root may use the future-only release.
    auto actual_collision=live;actual_collision.hard[80*160+80]=actual_collision.present_hard[80*160+80]=1;
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,actual_collision));
    auto wall=*map;Eigen::Array3i wall_index;
    ASSERT_TRUE(wall.grid.pos2Index(Eigen::Vector3d(1.45,.025,0.),wall_index));wall.grid.at("collision_risk",wall_index)=1.;
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,wall,live));
    auto unknown=*map;for(auto &slice:unknown.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,unknown,live));
    auto unowned=live;unowned.held_moving_cells.clear(); // Any unknown/static witness retains the cell.
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,unowned));
    auto stationary=live;stationary.tracks[0].gaussian.mean.tail<2>().setZero();stationary.tracks[0].vy=0.;
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,stationary));
    auto duplicate=live;duplicate.tracks.push_back(duplicate.tracks.front());
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,duplicate));
    auto unmatched=*map;unmatched.moving_subcells.clear();
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,commands,unmatched,live));
    auto stale=*map;stale.stamp=rclcpp::Time(map->stamp.nanoseconds()-10000000LL);stale.moving_subcells.clear();
    n->labelMovingSubcells(stale,live);EXPECT_TRUE(stale.moving_subcells.empty());
    // Exclusions are tied to the exact immutable grid, never a copied/new map.
    EXPECT_FALSE(n->collisionAtPoint(map->grid,index,1.025,.025,cache.current_view.get()));
    EXPECT_TRUE(n->collisionAtPoint(unmatched.grid,index,1.025,.025,cache.current_view.get()));
    EXPECT_TRUE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,live,nullptr,nullptr,&failure,nullptr,&cache));
  }
  static void earlier_stopping_guard_uses_the_configured_model_without_missing_crossing_times()
  {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";
    n->motion_joint_risk_limit_=.05;n->live_obstacle_inflation_radius_=0.;
    terrain_nav::YieldState root;terrain_nav::YieldModel model;model.wheelbase=.43;
    model.rear_axle_offset=.215;
    auto rest=terrain_nav::rolloutYield(root,root,{{0,0,0}},model,3.);
    LiveObstacleSnapshot live;terrain_nav::MotionTrackSnapshot t;
    t.id=7;t.motion_confirmed=t.circular=t.center_motion_fit=true;
    t.stamp=100.;t.resolution=.05;t.center_x=3.;t.center_y=-2.7;t.radius_bound=.55;
    t.cells={{{3.,-2.7}}};t.gaussian.valid=true;t.gaussian.stamp=100.;
    t.gaussian.observation_count=20;t.gaussian.mean<<3.,-2.7,0.,0.;
    t.gaussian.covariance=Eigen::Matrix4d::Identity()*.000324;
    t.gaussian.process_density=.04;t.acceleration_bound=1.2;
    live.tracks={t};
    EXPECT_TRUE(n->timedMovingUnsafe(terrain_nav::possibleEarlyStops(rest),100.,live)); // Old fallback.
    EXPECT_FALSE(n->earlierStoppingUnsafe(rest,100.,live)); // Same valid Gaussian event as stopping.
    // An actor passes the rest footprint at t=1.5, but is far at both ends.
    // Keep the WHOLE future interval, not just a single final-time point.
    t.center_x=0.;t.center_y=3.;t.vy=-2.;t.cells={{{0.,3.}}};
    t.gaussian.mean<<0.,3.,0.,-2.;t.gaussian.covariance.setZero();t.gaussian.process_density=0.;
    live.tracks={t};EXPECT_TRUE(n->earlierStoppingUnsafe(rest,100.,live));
    auto stale=live;stale.tracks[0].stamp-=1.;
    EXPECT_TRUE(n->earlierStoppingUnsafe(rest,100.,stale));
  }
  static void modeled_motion_speed_policy_never_grants_an_unsafe_command()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->dynamic_time_indexed_routing_=true;n->unclassified_encounter_speed_=.6;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=map->stamp.seconds();live->min_x=live->min_y=-4.;
    live->width=live->height=40;live->resolution=.2;live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot t;t.id=7;t.motion_confirmed=t.circular=t.center_motion_fit=true;
    t.stamp=live->stamp_sec;t.resolution=.05;t.center_x=.4;t.radius_bound=.45;
    t.cells={{{.2,-.2}},{{.6,-.2}},{{.2,.2}},{{.6,.2}}};t.gaussian.valid=true;
    t.gaussian.stamp=t.stamp;t.gaussian.observation_count=4;t.gaussian.mean<<.4,0.,0.,.8;
    t.gaussian.process_density=.04;t.vy=.8;live->tracks={t};live->compact_observed={{.175,-.225,.625,.225}};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0,0,0,*live),.6); // Default unchanged.
    n->unclassified_encounter_model_release_=true;
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0,0,0,*live),1.33);
    OdomState odom;odom.valid=true;odom.stamp=n->now();geometry_msgs::msg::Twist cmd;cmd.linear.x=1.33;
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd)); // Current body already intersects this actor.
    // Established identity now measured at rest: allow full proposal speed,
    // but never release its occupied body or its stochastic future motion.
    live->tracks[0].gaussian.mean.tail<2>().setZero();
    live->tracks[0].vx=live->tracks[0].vy=0.;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0,0,0,*live),1.33);
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
    auto stale=*live;stale.stamp_sec-=.7;
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0,0,0,stale),.6);
    n->dynamic_time_indexed_routing_=false;
    EXPECT_DOUBLE_EQ(n->encounterSpeed(1.33,0,0,0,*live),.6);
  }
  static void fair_family_proofs_keep_better_forward_motion_and_safe_budget_fallback()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    LiveObstacleSnapshot live;live.stamp_sec=map->stamp.seconds();live.min_x=live.min_y=-4.;
    live.width=live.height=40;live.resolution=.2;live.hard.assign(1600,0);live.present_hard=live.hard;
    Eigen::Array3i obstacle;ASSERT_TRUE(map->grid.pos2Index(Eigen::Vector3d(1.8,0.,0.),obstacle));
    map->grid.at("collision_risk",obstacle)=1.;
    terrain_nav::YieldModel model;model.wheelbase=n->wheel_base_;model.rear_axle_offset=n->rear_axle_offset_;
    terrain_nav::YieldState root;root.speed=.2;
    const std::vector<std::vector<terrain_nav::YieldCommand>> commands{
      {{0,.8,0.},{2.5,0.,0.}},{{0,0.,0.}},{{0,.4,0.},{1.,0.,0.}},{{0,-.3,0.},{1.,0.,0.}}};
    std::vector<std::vector<terrain_nav::YieldState>> states;
    for(const auto &c:commands)states.push_back(terrain_nav::rolloutYield(root,root,c,model,4.));
    const std::vector<double> utility{3.,0.,1.5,-1.};
    const auto order=terrain_nav::interactionBranchOrder({1,0,1,-1});
    EXPECT_EQ(order,(std::vector<size_t>{0,1,3,2}));
    const auto prove=[&](size_t i) {return n->timedYieldSafe(states[i],live.stamp_sec,commands[i],*map,live);};
    ASSERT_FALSE(prove(0));ASSERT_TRUE(prove(1));ASSERT_TRUE(prove(2));ASSERT_TRUE(prove(3));
    int checked=0;
    auto selected=terrain_nav::bestCertifiedInteraction(utility,order,[&](size_t i){++checked;return prove(i);},[] {return true;});
    EXPECT_EQ(selected,2u);EXPECT_EQ(checked,3); // Reverse is worse than the already-proved brake.
    // Budget exhaustion keeps only a proved fallback; invalid input or all
    // failed proofs cannot manufacture a command. No safety limit changes.
    checked=0;
    selected=terrain_nav::bestCertifiedInteraction(utility,order,[&](size_t i){++checked;return prove(i);},[&] {return checked<2;});
    EXPECT_EQ(selected,1u);
    EXPECT_EQ(terrain_nav::bestCertifiedInteraction(utility,order,[](size_t){return false;},[] {return true;}),std::numeric_limits<size_t>::max());
    auto invalid=utility;invalid[2]=NAN;
    EXPECT_EQ(terrain_nav::bestCertifiedInteraction(invalid,order,prove,[] {return true;}),std::numeric_limits<size_t>::max());
    const auto screened=terrain_nav::interactionProofOrder({1,0,1,-1},{.04,.02,.001,.03});
    checked=0;
    EXPECT_EQ(terrain_nav::bestCertifiedInteraction(utility,screened,[&](size_t i){++checked;return prove(i);},
      [&]{return checked<1;}),2u); // Real terrain/stop proof, one available slot.
    // Minimum screen estimate cannot release a real obstacle collision.
    const auto misleading=terrain_nav::interactionProofOrder({1,0,1,-1},{0.,.02,.01,.03});
    checked=0;
    EXPECT_EQ(terrain_nav::bestCertifiedInteraction(utility,misleading,[&](size_t i){++checked;return prove(i);},
      [&]{return checked<1;}),std::numeric_limits<size_t>::max());
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_EQ(terrain_nav::bestCertifiedInteraction(utility,order,prove,[] {return true;}),std::numeric_limits<size_t>::max());
  }
  static void decision_geometry_cache_preserves_policy_ledgers_and_input_generations()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    LiveObstacleSnapshot live;live.stamp_sec=map->stamp.seconds();live.min_x=live.min_y=-4.;
    live.width=live.height=40;live.resolution=.2;live.hard.assign(1600,0);live.present_hard=live.hard;
    YieldFootprintCache cache;
    terrain_nav::YieldModel model;model.wheelbase=n->wheel_base_;model.rear_axle_offset=n->rear_axle_offset_;
    for(double speed:{0.,.2,-.2}) {
      terrain_nav::YieldState root;root.speed=speed;
      const std::vector<terrain_nav::YieldCommand> commands{{0.,speed,0.},{.8,0.,0.}};
      const auto states=terrain_nav::rolloutYield(root,root,commands,model,3.);
      double path=0,stop=0,cached_path=0,cached_stop=0;std::string reason,cached_reason;
      ASSERT_TRUE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,live,&path,&stop,&reason));
      ASSERT_TRUE(n->timedYieldSafe(states,live.stamp_sec,commands,*map,live,&cached_path,&cached_stop,
                                   &cached_reason,nullptr,&cache));
      EXPECT_DOUBLE_EQ(path,cached_path);EXPECT_DOUBLE_EQ(stop,cached_stop);EXPECT_EQ(reason,cached_reason);
    }
    EXPECT_GT(cache.hits,100u);EXPECT_GT(cache.misses,0u);EXPECT_LT(cache.entries.size(),8192u);
    const std::vector<terrain_nav::YieldState> states{{0.,0.,0.,0.,0.,0.},{3.,0.,0.,0.,0.,0.}};
    auto unknown=*map;for(auto &slice:unknown.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,{{0,0,0}},unknown,live,nullptr,nullptr,nullptr,nullptr,&cache));
    auto occupied=live;occupied.hard[20*40+20]=occupied.present_hard[20*40+20]=1;
    EXPECT_FALSE(n->timedYieldSafe(states,live.stamp_sec,{{0,0,0}},*map,occupied,nullptr,nullptr,nullptr,nullptr,&cache));
    // Returning to the original generation must also clear rejected queries.
    EXPECT_TRUE(n->timedYieldSafe(states,live.stamp_sec,{{0,0,0}},*map,live,nullptr,nullptr,nullptr,nullptr,&cache));
  }
  static void retiring_bookkeeping_preserves_a_fresh_policy_during_nominal_recovery()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->reverse_recovery_enable_=n->reverse_enable_=true;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=.2;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.};
    n->dynamic_conflict_goal_=1;n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=100.;live->min_x=live->min_y=-4.;
    live->width=live->height=40;live->resolution=.2;live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot far;far.id=7;far.stamp=100.;far.resolution=.05;
    far.cells={{{-3.,1.}}};far.vx=-.3;far.vy=-.6;far.motion_confirmed=true;live->tracks={far};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->dynamic_unresolved_[7]=100.;n->dynamic_active_=true;
    n->dynamic_origin_=99.7;n->dynamic_last_time_=99.98;n->dynamic_goal_seq_=1;
    n->dynamic_epoch_=n->input_epoch_.load();n->dynamic_horizon_sec_=4.;
    n->dynamic_initial_={0.,0.,0.,0.,.2,0.};n->dynamic_commands_={{0.,.5,0.},{1.5,0.,0.}};
    n->stale_inputs_waiting_.store(true); // ordinary controller cannot take over yet
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_,true));
    EXPECT_TRUE(n->dynamic_unresolved_.empty());EXPECT_TRUE(n->dynamic_active_);
    EXPECT_EQ(n->dynamic_geometric_retirements_,1u);
    EXPECT_DOUBLE_EQ(n->dynamic_origin_,99.7);EXPECT_EQ(n->dynamic_commands_.size(),2u);
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,.5);EXPECT_TRUE(n->last_chance_accepted_);
    // Preserving an executing policy is not permission across a new goal.
    ++n->goal_seq_;
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_,true));
    EXPECT_FALSE(n->dynamic_active_);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
  }
  static void footprint_query_reuse_keeps_the_same_pose_and_height_ledgers()
  {
    for(bool source_ledger:{false,true}) {
      auto n=node(.002F);
      if(source_ledger) {
        n->map_["probability_contract"][0].setConstant(5.);
        n->map_.add("probability_pose_event",true,0.);
        n->map_.add("height_error_tail_upper",true,.0001F);
        n->map_.add("probability_height_radius_cells",false,6.);
      }
      auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      snapshot->grid=n->map_;
      terrain_nav::ChanceBudget original,reused;
      for(int i=0;i<40;++i) {
        const double x=-1.+.05*i,y=.1*std::sin(i*.2),yaw=.07*i;
        double cost=0,clearance=0;CostQuery query;
        ASSERT_TRUE(n->footprintCostLocked(x,y,yaw,cost,&clearance,nullptr,false,snapshot.get(),nullptr,&query));
        EXPECT_EQ(query.probability_upper,n->queryCostLocked(x,y,yaw,false,snapshot.get()).probability_upper);
        EXPECT_EQ(n->addChancePose(x,y,yaw,original,.05,snapshot.get()),
                  n->addChanceQuery(query,reused,.05,snapshot.get()));
        EXPECT_DOUBLE_EQ(original.upper(),reused.upper());
        if(source_ledger)EXPECT_GT(query.height_radius,0);
      }
      // A new generation performs its own footprint/query; neither a cached
      // prior certificate nor a failed footprint can supply a valid query.
      auto unsafe=*snapshot;
      for(auto &slice:unsafe.grid["probability_valid"])slice.setZero();
      CostQuery query;double cost=0;terrain_nav::ChanceBudget rejected;
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost,nullptr,nullptr,false,&unsafe,nullptr,&query));
      EXPECT_FALSE(n->addChanceQuery(query,rejected,.05,&unsafe));
      auto unknown=*snapshot;unknown.grid["collision_risk"][0].setConstant(NAN);
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost,nullptr,nullptr,false,&unknown,nullptr,&query));
      EXPECT_FALSE(n->addChanceQuery(query,rejected,.05,&unknown));
    }
  }
  static void ordinary_braking_shares_the_combined_budget_and_checks_the_whole_tail()
  {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->dynamic_time_indexed_routing_=true;n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    const double stamp=n->now().seconds();
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4.;live->resolution=.2;
    live->width=live->height=40;live->stamp_sec=stamp;live->hard.assign(1600,0);live->present_hard=live->hard;
    const auto body=n->collisionEnvelope(n->footprint_margin_);
    terrain_nav::MotionTrackSnapshot t;t.circular=t.center_motion_fit=true;t.stamp=stamp;t.resolution=.05;t.radius_bound=.45;
    t.center_x=body.front+.45+.09+.38;t.center_y=0.;t.cells={{{t.center_x,0.}}};
    t.gaussian.valid=true;t.gaussian.stamp=stamp;t.gaussian.process_density=0.;t.gaussian.mean<<t.center_x,0.,0.,0.;
    t.gaussian.covariance(0,0)=.04;live->tracks={t};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    OdomState odom;odom.valid=true;odom.stamp=n->now();geometry_msgs::msg::Twist cmd;double upper=NAN;
    ASSERT_TRUE(n->commandBrakingEnvelopeSafe(odom,cmd,&upper));EXPECT_GT(upper,.02);EXPECT_LT(upper,.05);
    // A dynamic sub-budget may use unspent state risk, but the SUM is still
    // constrained by the unchanged 5% state/braking and 10% path limits.
    n->motion_joint_risk_limit_=.02;EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
    n->motion_joint_risk_limit_=.05;
    auto risky=std::make_shared<ControlMapSnapshot>(*map);
    for(auto &slice:risky->grid["failure_probability_upper"])slice.setConstant(.03F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(risky));
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd,&upper));EXPECT_GT(upper,.05);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    // Crossing occurs between the short braking motion and t=6. Endpoint
    // checks alone cannot prove the stationary tail clear under a CV model.
    auto crossing=std::make_shared<LiveObstacleSnapshot>(*live);t.center_x=0;t.center_y=3.;t.vy=-1.5;
    t.gaussian.mean<<0.,3.,0.,-1.5;t.gaussian.covariance.setZero();t.cells={{{0.,3.}}};crossing->tracks={t};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(crossing));
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
  }
  static void joint_motion_probability_is_part_of_terrain_admission()
  {
    auto n=node(1e-8F);n->motion_prediction_model_="gaussian_joint_cpu";
    n->motion_joint_risk_limit_=.05; // Same explicit 5% sub-budget as this simulation profile.
    const double time=n->now().seconds();
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    LiveObstacleSnapshot live;live.min_x=live.min_y=-4.;live.resolution=.2;live.width=live.height=40;
    live.stamp_sec=time;live.hard.assign(1600,0);live.present_hard=live.hard;
    terrain_nav::MotionTrackSnapshot actor;actor.circular=actor.center_motion_fit=true;
    actor.center_x=actor.center_y=3.;actor.radius_bound=.45;actor.resolution=.05;actor.stamp=time;
    actor.cells={{{3.,3.}}};actor.gaussian.valid=true;actor.gaussian.stamp=time;
    actor.gaussian.process_density=0.;actor.gaussian.mean<<3.,3.,0.,0.;
    live.tracks={actor};
    const std::vector<terrain_nav::YieldState> states={{0,0,0,0,0,0},{.02,0,0,0,0,0}};
    double path=NAN,brake=NAN;std::string failure;
    ASSERT_TRUE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live,&path,&brake,&failure))<<failure;
    EXPECT_GT(path,0.);EXPECT_GT(brake,0.);EXPECT_LT(path,.02);
    // A deterministic remote actor has zero analytic collision probability;
    // finite-sample confidence slack must not manufacture a risk event.
    for(auto &slice:map->grid["failure_probability_upper"])slice.setConstant(.0495F);
    EXPECT_TRUE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live,&path,&brake,&failure));
    // Use a genuinely uncertain NEAR actor to test addition of the two
    // event bounds, rather than relying on a zero-hit sampling floor.
    const auto body=n->collisionEnvelope(n->footprint_margin_);
    actor.center_x=body.front+.45+n->live_obstacle_inflation_radius_+.09+.38;
    actor.center_y=0.;actor.cells={{{actor.center_x,0.}}};
    actor.gaussian.mean<<actor.center_x,0.,0.,0.;actor.gaussian.covariance(0,0)=.04;
    live.tracks={actor};
    for(auto &slice:map->grid["failure_probability_upper"])slice.setConstant(1e-8F);
    ASSERT_TRUE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live,&path,&brake,&failure))<<failure;
    EXPECT_GT(path,.001);EXPECT_LT(path,.05);
    // Ground alone is below its 5% state limit, but ground + the separate
    // conditional dynamic upper bound must NOT be silently admitted.
    map->grid["failure_probability_upper"][0].setConstant(.0495F);
    for(auto &slice:map->grid["failure_probability_upper"])slice.setConstant(.0495F);
    EXPECT_FALSE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live,&path,&brake,&failure));
    EXPECT_EQ(failure,"combined terrain/dynamic state probability");
    for(auto &slice:map->grid["failure_probability_upper"])slice.setConstant(1e-8F);
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live));
    for(auto &slice:map->grid["probability_valid"])slice.setOnes();
    live.tracks[0].stamp=time-1.;live.tracks[0].gaussian.stamp=time-1.;
    EXPECT_FALSE(n->timedYieldSafe(states,time,{{0,0,0}},*map,live,&path,&brake,&failure));
    EXPECT_EQ(failure,"conditional joint motion risk");
    EXPECT_TRUE(std::isnan(path));
    // The early-stop PRIORITY locus repeats a future time. Treating it as
    // an executed Gaussian policy would miss an actor that crossed earlier.
    actor.stamp=actor.gaussian.stamp=time;actor.center_x=0;actor.center_y=3.;
    actor.vx=0;actor.vy=-1.5;actor.gaussian.mean<<0.,3.,0.,-1.5;
    actor.cells={{{0.,3.}}};live.tracks={actor};
    const std::vector<terrain_nav::YieldState> locus={{0,0,0,0,0,0},{4.,0,0,0,0,0},{4.,0,0,0,0,0}};
    EXPECT_TRUE(n->timedMovingUnsafe(locus,time,live));
  }
  static void bounded_clearance_matches_unbounded_octile_distances()
  {
    auto n=node(1e-8F);ControlMapSnapshot snapshot;
    snapshot.width=51;snapshot.height=41;snapshot.resolution=.2;
    snapshot.min_x=-5.1;snapshot.min_y=-4.1;
    snapshot.grid.initGeometry(Eigen::Array2d(10.2,8.2),.2,2*M_PI/36,Eigen::Vector2d::Zero(),2*M_PI);
    snapshot.grid.add("historical_obstacle_advisory",false,0.);
    const int count=snapshot.width*snapshot.height;
    snapshot.hard.assign(count,0);snapshot.costs.assign(count,0);
    snapshot.clearance.resize(count);snapshot.comfort_clearance.resize(count);
    const std::vector<std::array<int,2>> hard={{{0,0}},{{25,20}},{{50,40}}};
    auto comfort=hard;comfort.push_back({7,3});
    const std::vector<std::array<int,2>> history={{{40,9}}};
    for(const auto &p:hard){snapshot.hard[p[1]*51+p[0]]=1;snapshot.costs[p[1]*51+p[0]]=254;}
    snapshot.costs[3*51+7]=100;
    Eigen::Array3i index;
    ASSERT_TRUE(snapshot.grid.pos2Index(Eigen::Vector3d(-5.1+40.5*.2,-4.1+9.5*.2,0),index));
    snapshot.grid.at("historical_obstacle_advisory",index)=1.;
    n->computeClearance(snapshot);
    const auto exact=[&](int x,int y,const auto &seeds,double cap) {
      double result=cap;
      for(const auto &p:seeds) {
        const int dx=std::abs(x-p[0]),dy=std::abs(y-p[1]);
        result=std::min(result,.2*(std::max(dx,dy)+(std::sqrt(2.)-1)*std::min(dx,dy)));
      }
      return result;
    };
    for(int y=0;y<41;++y)for(int x=0;x<51;++x) {
      EXPECT_NEAR(snapshot.clearance[y*51+x],exact(x,y,hard,n->clearance_distance_+2),1e-6);
      EXPECT_NEAR(snapshot.comfort_clearance[y*51+x],exact(x,y,comfort,n->comfort_clearance_distance_+2),1e-6);
      EXPECT_NEAR(snapshot.historical_clearance[y*51+x],exact(x,y,history,n->historical_obstacle_influence_distance_),1e-6);
    }
    snapshot.hard.assign(count,0);snapshot.costs.assign(count,0);
    snapshot.grid["historical_obstacle_advisory"][0].setZero();n->computeClearance(snapshot);
    for(float value:snapshot.clearance)EXPECT_FLOAT_EQ(value,float(n->clearance_distance_+2));
  }
  static void live_witness_geometry_keeps_the_full_body_and_prediction_fallback()
  {
    auto n=node(1e-8F);n->body_collision_front_=.58;n->body_collision_back_=.48;
    n->body_collision_half_width_=.40;n->footprint_margin_=.10;
    const double x=8.13386111433546,y=.6190615371919107,yaw=.5383006602506807;
    LiveObstacleSnapshot live;live.min_x=8.15;live.min_y=0.;live.resolution=.05;
    live.width=live.height=1;live.hard={1};live.present_hard={1};live.uninflated_present={1};
    live.forecast={0};live.inflation_radius=0.;
    ASSERT_EQ(n->liveBodyOverlapCount(x,y,yaw,live),1u); // whole 5cm square
    const std::vector<std::array<float,3>> returns={
      {8.199299F,.001671285F,.427671F},{8.192813F,.000398747F,.454525F},
      {8.193688F,.001784377F,.656715F},{8.199168F,.001332173F,.649322F},
      {8.194981F,.001549200F,.765881F},{8.199957F,.001797025F,.508232F},
      {8.197440F,.001031516F,.902294F},{8.196871F,.000356237F,.432102F}};
    live.rectangles=terrain_nav::columnWitnessRectangles(returns,live.hard,1,1,8.15,0.,.05);
    ASSERT_TRUE(live.rectangles[0].valid());
    EXPECT_EQ(n->liveBodyOverlapCount(x,y,yaw,live),0u);
    EXPECT_FALSE(live.occupied(8.175,.045)); // same geometry for point queries
    EXPECT_TRUE(live.occupied(8.195,.001));
    EXPECT_EQ(n->liveBodyOverlapCount(x+.05*std::sin(yaw),y-.05*std::cos(yaw),yaw,live),1u);
    // Forecasts and malformed/missing evidence retain the original square.
    live.forecast[0]=1;EXPECT_EQ(n->liveBodyOverlapCount(x,y,yaw,live),1u);
    live.forecast[0]=0;live.rectangles[0].x0=NAN;
    EXPECT_EQ(n->liveBodyOverlapCount(x,y,yaw,live),1u);
    live.rectangles.clear();EXPECT_EQ(n->liveBodyOverlapCount(x,y,yaw,live),1u);
    const auto envelope=n->collisionEnvelope(.1);
    EXPECT_NEAR(envelope.front,.68,1e-12);EXPECT_NEAR(envelope.back,.58,1e-12);EXPECT_DOUBLE_EQ(envelope.half_width,.5);
  }

  static void live_witness_history_obeys_the_original_hold_and_does_not_grow_forever()
  {
    auto n=node(1e-8F);n->live_obstacle_resolution_=.05;n->live_obstacle_min_hits_=1;
    n->live_obstacle_inflation_radius_=0.;n->live_obstacle_hold_sec_=.3;n->live_motion_guard_enable_=false;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    snapshot->stamp=n->now();snapshot->grid.add("terrain_observed",false,1.F);
    snapshot->grid.add("gp_elevation_observed",false,0.F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    const auto submit=[&](double time,float value,bool has_point=true) {
      EXPECT_EQ(rcl_set_ros_time_override(clock,int64_t(std::llround(time*1e9))),RCL_RET_OK);
      auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();cloud->header.frame_id="map";cloud->header.stamp=n->now();
      sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(has_point?1:0);
      if(has_point) {
        sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
        *x=value;*y=value;*z=.4F;
      }
      n->cloudCallback(cloud);
      return std::atomic_load(&n->live_snapshot_);
    };
    auto first=submit(100.,1.201F);ASSERT_TRUE(first);EXPECT_TRUE(first->occupied(1.201,1.201));
    auto both=submit(100.1,1.249F);ASSERT_TRUE(both);EXPECT_TRUE(both->occupied(1.201,1.201));
    auto recent=submit(100.31,1.249F);ASSERT_TRUE(recent);
    EXPECT_FALSE(recent->occupied(1.201,1.201));EXPECT_TRUE(recent->occupied(1.249,1.249));
    auto held=submit(100.4,0.F,false);ASSERT_TRUE(held);EXPECT_TRUE(held->occupied(1.249,1.249));
    auto expired=submit(100.7,0.F,false);ASSERT_TRUE(expired);EXPECT_FALSE(expired->occupied(1.249,1.249));
  }
  static void offset_body_and_rear_tracking_agree_in_both_gears()
  {
    auto n=node(1e-8F);n->rear_axle_offset_=.2078;n->live_obstacle_enable_=false;
    for(int direction:{-1,1})for(double k:{-.7,-.4,.4,.7}) {
      std::vector<Pose2> path,wrong;
      for(int i=0;i<=120;++i) {
        auto p=n->advanceArc(Pose2{},k,direction*i*.02);p.s=i*.02;p.direction=direction;path.push_back(p);
        p=::advanceArc(Pose2{},k,direction*i*.02);p.s=i*.02;p.direction=direction;wrong.push_back(p);
      }
      ASSERT_TRUE(n->bodyArcPathFeasible(path));EXPECT_FALSE(n->bodyArcPathFeasible(wrong));
      OdomState odom;odom.valid=true;odom.v=.3*direction;
      for(double weight:{0.,.45,1.}) {
        n->pure_pursuit_weight_=weight;
        const auto cmd=n->computePurePursuitCommand(odom,path);
        EXPECT_NEAR(cmd.angular.z,std::atan(n->wheel_base_*k),3e-4);
        EXPECT_GT(direction*cmd.linear.x,0.);
      }
      const auto rebuilt=n->reanchorArc(path,odom);
      ASSERT_FALSE(rebuilt.empty());
      if(direction>0){EXPECT_NEAR(rebuilt.back().x,path.back().x,2e-4);EXPECT_NEAR(rebuilt.back().y,path.back().y,2e-4);}
    }
  }
  static void offset_connectors_keep_base_endpoints_and_full_body_heading()
  {
    auto n=node(1e-8F);n->rear_axle_offset_=.2078;n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    for(int direction:{-1,1})for(double k:{-.4,.4}) {
      Pose2 start{};auto goal=n->advanceArc(start,k,direction*1.5);std::vector<Pose2> path;
      const bool ok=direction>0?n->buildArcDirectPathLocked(start,goal,path):n->buildCertifiedReverseArcLocked(start,goal,path);
      ASSERT_TRUE(ok);EXPECT_TRUE(n->bodyArcPathFeasible(path));
      EXPECT_NEAR(path.back().x,goal.x,1e-5);EXPECT_NEAR(path.back().y,goal.y,1e-5);
      EXPECT_NEAR(wrapAngle(path.back().yaw-goal.yaw),0.,1e-5);
      ASSERT_TRUE(n->buildFrenetDirectPathLocked(start,n->advanceArc(start,k,1.5),path));
      EXPECT_TRUE(n->bodyArcPathFeasible(path));
    }
  }
  static void varying_curvature_reference_has_correct_feedforward_in_each_gear()
  {
    auto n=node(1e-8F);n->rear_axle_offset_=.2078;n->live_obstacle_enable_=false;
    n->terrain_speed_enable_=false;
    for(int gear:{-1,1}) {
      Pose2 p{};p.direction=gear;std::vector<Pose2> path{p};std::vector<double> edge_k;
      for(int i=0;i<160;++i) {
        const double k=(i/12)%3==0?.68:(i/12)%3==1?-.55:.12;
        p=n->advanceArc(p,k,gear*.025);p.s=(i+1)*.025;p.direction=gear;
        path.push_back(p);edge_k.push_back(k);
      }
      ASSERT_TRUE(n->bodyArcPathFeasible(path));
      for(double weight:{0.,.42,1.}) {
        n->pure_pursuit_weight_=weight;
        for(size_t i=1;i+2<path.size();++i) {
          OdomState actual;actual.x=path[i].x;actual.y=path[i].y;actual.yaw=path[i].yaw;actual.v=gear*.3;
          const auto cmd=n->computePurePursuitCommand(actual,path);
          EXPECT_NEAR(cmd.angular.z,std::atan(n->wheel_base_*edge_k[i]),1e-7)
            << "gear=" << gear << " edge=" << i << " blend=" << weight;
        }
      }
    }
  }
  static void offset_staged_plan_recertifies_reverse_then_forward()
  {
    auto n=staged_fixture(.2078);
    for(int expected:{-1,1}) {
      const auto end=n->path_.back();n->odom_.x=end.x;n->odom_.y=end.y;n->odom_.yaw=end.yaw;
      n->odom_.v=0.;n->odom_.stamp=n->now();
      ASSERT_TRUE(n->planPath("offset staged cusp"));EXPECT_EQ(n->path_.front().direction,expected);
      EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    }
  }
  static void staged_execution_keeps_uncertified_future_tail_without_executing_it()
  {
    auto n=staged_fixture(.2078);n->path_={{0.,0.,0.,0.,-1},{-.1,0.,0.,.1,-1}};
    n->recovery_legs_.clear();std::vector<Pose2> forward;
    for(int i=0;i<=20;++i)forward.push_back({-.1+.1*i,0.,0.,.1*i,1});
    n->recovery_legs_.push_back(forward);
    n->odom_.x=-.1;n->odom_.y=0.;n->odom_.yaw=0.;n->odom_.v=0.;n->odom_.stamp=n->now();
    for(auto &slice:n->map_["probability_valid"])
      for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()>.7)slice(r,c)=0.;
      }
    ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,1);EXPECT_LT(n->path_.back().x,1.);
    ASSERT_EQ(n->recovery_legs_.size(),1u);
    EXPECT_NEAR(n->recovery_legs_.front().front().x,n->path_.back().x,1e-8);
    EXPECT_NEAR(n->recovery_legs_.front().back().x,1.9,1e-8);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    EXPECT_FALSE(n->chancePathSafe(n->recovery_legs_.front(),nullptr,.1));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    n->odom_.x=n->path_.back().x;n->odom_.stamp=n->now();
    ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,1);EXPECT_NEAR(n->path_.back().x,1.9,1e-8);
    EXPECT_TRUE(n->recovery_legs_.empty());
  }
  static void short_certified_cusp_remainder_is_not_an_unknown_observation_prefix()
  {
    for(int gear:{-1,1}) {
      auto n=staged_fixture(.2078);n->path_={{-gear*.1,0.,0.,0.,-gear},{0.,0.,0.,.1,-gear}};
      const std::vector<Pose2> tail={{0.,0.,0.,0.,gear},{gear*.05,0.,0.,.05,gear},{gear*.1,0.,0.,.1,gear}};
      n->recovery_legs_={tail};n->odom_.x=0.;n->odom_.y=.02;n->odom_.yaw=0.;n->odom_.v=0.;
      std::vector<Pose2> prefix;
      EXPECT_FALSE(n->buildCertifiedGuidePrefixLocked(tail,prefix)); // still too short for an observation frontier
      ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(tail,prefix,true)); // wholly certified cusp remainder
      const auto reference=n->joinFixedGearLegLocked(tail,n->odom_);
      ASSERT_EQ(reference.size(),tail.size());
      for(size_t i=0;i<tail.size();++i) {
        EXPECT_DOUBLE_EQ(reference[i].x,tail[i].x);EXPECT_DOUBLE_EQ(reference[i].y,tail[i].y);
        EXPECT_DOUBLE_EQ(reference[i].yaw,tail[i].yaw);
      }
      ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
      EXPECT_EQ(n->path_.front().direction,gear);EXPECT_NEAR(n->path_.back().x,gear*.1,1e-9);
      EXPECT_TRUE(n->recovery_legs_.empty());
      EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    }
    auto n=staged_fixture();std::vector<Pose2> tail={{0,0,0,0,1},{.1,0,0,.1,1}};
    OdomState pose;pose.y=.02;pose.valid=true;
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    std::vector<Pose2> prefix;EXPECT_FALSE(n->buildCertifiedGuidePrefixLocked(tail,prefix,true));
    EXPECT_TRUE(n->joinFixedGearLegLocked(tail,pose).empty());
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["collision_risk"])slice.setOnes();
    EXPECT_TRUE(n->joinFixedGearLegLocked(tail,pose).empty());
    for(auto &slice:n->map_["collision_risk"])slice.setZero();
    pose.y=.06;EXPECT_TRUE(n->joinFixedGearLegLocked(tail,pose).empty()); // outside capture: need a real connector
  }
  static void unsupported_reverse_cost_is_finite_and_never_changes_admission()
  {
    auto n=node(1e-8F);n->topology_unknown_reverse_weight_=3.;n->hard_clearance_floor_=0.;
    EXPECT_EQ(n->topologyReverseObservationCostLocked(0.,0.,0.),0.);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    const double penalty=n->topologyReverseObservationCostLocked(0.,0.,0.);
    EXPECT_TRUE(std::isfinite(penalty));EXPECT_GT(penalty,0.);
    EXPECT_FALSE(n->queryCostLocked(0.,0.,0.).known);
    terrain_nav::ChanceBudget budget;EXPECT_FALSE(n->addChancePose(0.,0.,0.,budget,.05));
    n->topology_unknown_reverse_weight_=0.;
    EXPECT_EQ(n->topologyReverseObservationCostLocked(0.,0.,0.),0.);
    EXPECT_FALSE(n->queryCostLocked(0.,0.,0.).known);
  }
  static void complete_forward_seed_requires_local_certificate_but_future_is_only_guidance()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->hard_clearance_floor_=0.;
    n->live_obstacle_enable_=false;OdomState odom;odom.valid=true;
    std::vector<terrain_nav::RouteMemory::Point> floor;
    for(int y=-35;y<=35;++y)for(int x=-35;x<=35;++x)floor.push_back({x*.1,y*.1});
    n->route_memory_.observe(1.,{-4,-4,4,4},floor,{});
    terrain_nav::RouteMemory::Result seed;
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,Pose2{3.,0.,0.},seed));
    ASSERT_TRUE(seed.reached);for(int gear:seed.directions)EXPECT_EQ(gear,1);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->completeForwardSeedLocked(odom,Pose2{3.,0.,0.},seed));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    // A high future UPPER bound is not proof of an obstacle. Admit only
    // geometric intent with a fresh safe prefix, NEVER the whole route.
    for(auto &slice:n->map_["failure_probability_upper"])
      for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()>1.5)slice(r,c)=.9;
      }
    EXPECT_TRUE(n->queryCostLocked(0.,0.,0.).known);
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,Pose2{3.,0.,0.},seed));
    std::vector<Pose2> guide,certificate;double length=0.;
    for(size_t i=0;i<seed.path.size();++i) {
      if(i)length+=std::hypot(seed.path[i][0]-seed.path[i-1][0],seed.path[i][1]-seed.path[i-1][1]);
      guide.push_back({seed.path[i][0],seed.path[i][1],seed.yaws[i],length});
    }
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,certificate));
    EXPECT_LT(certificate.back().x,1.6);
    EXPECT_TRUE(n->chancePathSafe(certificate,nullptr,.1));
    EXPECT_FALSE(n->chancePathSafe(guide,nullptr,.1));
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(.9F);
    EXPECT_FALSE(n->completeForwardSeedLocked(odom,Pose2{3.,0.,0.},seed));
  }
  static void retained_forward_reference_recertifies_after_time_epoch_but_never_space_or_hazard()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->hard_clearance_floor_=0.;
    n->live_obstacle_enable_=false;OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
    std::vector<terrain_nav::RouteMemory::Point> floor;
    for(int y=-35;y<=35;++y)for(int x=-35;x<=35;++x)floor.push_back({x*.1,y*.1});
    n->route_memory_.observe(1.,{-4,-4,4,4},floor,{});
    terrain_nav::RouteMemory::Result seed;
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,goal,seed));
    const auto near=seed.path.size()/6;odom.x=seed.path[near][0];odom.y=seed.path[near][1];odom.yaw=seed.yaws[near];
    ++n->input_epoch_; // only the geometric reference survives a temporal gap
    ASSERT_TRUE(n->reuseForwardSeedLocked(odom,goal,seed));EXPECT_EQ(seed.expanded,0);
    EXPECT_NEAR(seed.path.front()[0],odom.x,1e-8);
    auto failed=std::make_shared<std::vector<TerrainNavNode::RecoveryRouteMemory>>();
    failed->push_back({1.5,0.,n->now().seconds(),n->input_epoch_.load()});
    std::atomic_store(&n->recovery_route_memory_,std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>(failed));
    EXPECT_FALSE(n->reuseForwardSeedLocked(odom,goal,seed));
    EXPECT_TRUE(std::isfinite(n->recoveryRoutePenalty(1.5,0.)));
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,goal,seed)); // finite preference is not a route ban
    // Simulate adoption after the same current finite preference was scored.
    // Crossing its region cannot categorically invalidate the fresh guide.
    auto straight=seed;straight.path={{odom.x,odom.y},{1.5,0.},{3.,0.}};
    straight.yaws={0.,0.,0.};straight.directions={1,1,1};n->forward_seed_=straight;
    EXPECT_TRUE(n->reuseForwardSeedLocked(odom,goal,seed));
    std::atomic_store(&n->recovery_route_memory_,std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>{});
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,goal,seed));
    auto changed=goal;changed.y+=.1;EXPECT_FALSE(n->reuseForwardSeedLocked(odom,changed,seed));
    auto displaced=odom;displaced.y+=.2;EXPECT_FALSE(n->reuseForwardSeedLocked(displaced,goal,seed));
    ++n->spatial_epoch_;EXPECT_FALSE(n->reuseForwardSeedLocked(odom,goal,seed));
    --n->spatial_epoch_;
    for(auto &slice:n->map_["probability_valid"])
      for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()>1.5)slice(r,c)=0.;
      }
    ASSERT_TRUE(n->reuseForwardSeedLocked(odom,goal,seed)); // geometry can remain guidance beyond unknown support
    std::vector<Pose2> guide,certificate;double length=0.;
    for(size_t i=0;i<seed.path.size();++i) {
      if(i)length+=std::hypot(seed.path[i][0]-seed.path[i-1][0],seed.path[i][1]-seed.path[i-1][1]);
      guide.push_back({seed.path[i][0],seed.path[i][1],seed.yaws[i],length});
    }
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,certificate));
    EXPECT_LT(certificate.back().x,1.6);EXPECT_LT(certificate.back().s,guide.back().s);
    EXPECT_TRUE(n->chancePathSafe(certificate,nullptr,.1));EXPECT_FALSE(n->chancePathSafe(guide,nullptr,.1));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(.9F);
    EXPECT_FALSE(n->reuseForwardSeedLocked(odom,goal,seed));
  }
  static void complete_seed_can_guide_beyond_a_short_certified_observation_prefix()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->hard_clearance_floor_=0.;
    n->live_obstacle_enable_=false;OdomState odom;odom.valid=true;
    std::vector<terrain_nav::RouteMemory::Point> floor;
    for(int y=-35;y<=35;++y)for(int x=-35;x<=35;++x)floor.push_back({x*.1,y*.1});
    n->route_memory_.observe(1.,{-4,-4,4,4},floor,{});
    for(auto &slice:n->map_["probability_valid"])
      for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()>.65)slice(r,c)=0.;
      }
    terrain_nav::RouteMemory::Result seed;ASSERT_TRUE(n->completeForwardSeedLocked(odom,Pose2{3.,0.,0.},seed));
    std::vector<Pose2> guide,certificate;double length=0.;
    for(size_t i=0;i<seed.path.size();++i) {
      if(i)length+=std::hypot(seed.path[i][0]-seed.path[i-1][0],seed.path[i][1]-seed.path[i-1][1]);
      guide.push_back({seed.path[i][0],seed.path[i][1],seed.yaws[i],length});
    }
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,certificate));
    EXPECT_GE(certificate.back().s,.25);EXPECT_LT(certificate.back().s,.75);
    EXPECT_TRUE(n->chancePathSafe(certificate,nullptr,.1));EXPECT_FALSE(n->chancePathSafe(guide,nullptr,.1));
  }
  static void background_results_are_only_guidance_and_require_current_identity_and_support()
  {
    for(int fault=0;fault<6;++fault) {
      auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->hard_clearance_floor_=0.;
      n->live_obstacle_enable_=false;OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
      std::vector<terrain_nav::RouteMemory::Point> floor;
      for(int y=-35;y<=35;++y)for(int x=-35;x<=35;++x)floor.push_back({x*.1,y*.1});
      n->route_memory_.observe(1.,{-4,-4,4,4},floor,{});
      terrain_nav::RouteMemory::Result route;ASSERT_TRUE(n->completeForwardSeedLocked(odom,goal,route));
      TerrainNavNode::BackgroundGuideResult result;result.route=route;result.goal=goal;
      result.spatial=n->spatial_epoch_.load();result.source_stamp=n->now().seconds();
      if(fault==1)++result.spatial;
      if(fault==2)result.goal.y+=1.;
      if(fault==3)result.source_stamp-=3.;
      if(fault==4)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      if(fault==5) {
        auto changed=std::make_shared<std::vector<TerrainNavNode::RecoveryRouteMemory>>();
        changed->push_back({1.5,0.,n->now().seconds(),n->input_epoch_.load()});
        std::atomic_store(&n->recovery_route_memory_,std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>(changed));
      }
      std::promise<TerrainNavNode::BackgroundGuideResult> promise;n->background_guide_=promise.get_future();
      promise.set_value(std::move(result));
      terrain_nav::RouteMemory::Result selected;
      EXPECT_EQ(n->takeBackgroundGuideLocked(odom,goal,selected),fault==0);
      EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty()); // adoption never publishes/executes a trajectory
      EXPECT_FALSE(n->background_guide_.valid());
    }
  }
  static void background_search_uses_immutable_copies_but_adoption_uses_new_support()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
    n->background_guide_runtime_=true;n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    n->map_stamp_=n->now();OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
    std::vector<terrain_nav::RouteMemory::Point> floor;
    for(int y=-35;y<=35;++y)for(int x=-35;x<=35;++x)floor.push_back({x*.1,y*.1});
    n->route_memory_.observe(1.,{-4,-4,4,4},floor,{});
    std::vector<float> lower(n->width_*n->height_,1.);terrain_nav::RouteOptions options;options.heuristic_weight=2.5;
    n->queueBackgroundGuideLocked(odom,goal,lower,options);ASSERT_TRUE(n->background_guide_.valid());
    // These would race and alter the result if any mutable node reference
    // had leaked into the background collision/terrain callbacks.
    n->route_memory_.clear();
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    auto result=n->background_guide_.get();ASSERT_TRUE(result.route.reached);
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
    std::promise<TerrainNavNode::BackgroundGuideResult> promise;n->background_guide_=promise.get_future();
    promise.set_value(std::move(result));terrain_nav::RouteMemory::Result selected;
    EXPECT_FALSE(n->takeBackgroundGuideLocked(odom,goal,selected));
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
  }
  static void mean_safe_uncertainty_is_not_costlier_guidance_than_wholly_unknown_ground()
  {
    auto n=node(.4F);n->comfort_clearance_weight_=5.;n->comfort_clearance_distance_=1.35;
    n->comfort_clearance_.assign(n->width_*n->height_,.2F);
    const double unknown=n->topologyTerrainCostLocked(100.,100.,0.);
    ASSERT_GT(n->topologyTerrainCostLocked(0.,0.,0.),unknown); // recorded legacy inversion
    n->map_.add("height_error_radius_m",true,.028F);
    const auto map_before=n->map_["failure_probability_upper"];
    EXPECT_LE(n->topologyTerrainCostLocked(0.,0.,0.),unknown);
    for(double comfort:{0.,2.5,5.}) {
      double previous=-1.;
      for(double p:{0.,.001,.01,.04,.05,.06,.4,1.}) {
        const double cost=TerrainNavNode::terrainGuidanceCost(p,.028,.05,5.,comfort);
        EXPECT_GE(cost,previous);EXPECT_LE(cost,unknown);previous=cost;
      }
      EXPECT_GT(TerrainNavNode::terrainGuidanceCost(1.,0.,.05,5.,comfort),unknown);
      EXPECT_GT(TerrainNavNode::terrainGuidanceCost(.4,NAN,.05,5.,comfort),unknown);
    }
    // A positive MEAN margin is never a terrain execution certificate. The
    // exact same over-budget pose and path remain inadmissible after routing
    // costs improve, and none of the recorded probability cells are changed.
    terrain_nav::ChanceBudget budget;
    EXPECT_FALSE(n->addChancePose(0.,0.,0.,budget,n->chance_state_limit_));
    std::vector<Pose2> path={{0.,0.,0.,0.,1},{.1,0.,0.,.1,1}};
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,n->chance_path_budget_));
    for(size_t k=0;k<map_before.size();++k)EXPECT_TRUE((map_before[k].array()==n->map_["failure_probability_upper"][k].array()).all());
    for(auto &slice:n->map_["height_error_radius_m"])slice.setZero();
    EXPECT_GT(n->topologyTerrainCostLocked(0.,0.,0.),unknown); // mean violation not discounted
  }

  static void background_candidate_ranking_keeps_risk_even_for_geometric_proposal()
  {
    // Quadrature unit test only: these polylines are not car certificates.
    terrain_nav::RouteMemory::Result short_route,detour;
    short_route.reached=detour.reached=true;
    short_route.path={{0,0},{1,0},{2,0},{3,0}};
    detour.path={{0,0},{1,1},{2,1},{3,0}};
    short_route.yaws=detour.yaws=std::vector<double>(4,0.);
    const auto length_cost=[](double,double,double){return 1.;};
    const auto risk_cost=[](double x,double y,double){return x>.5&&x<2.5&&y<.5?20.:1.;};
    EXPECT_LT(TerrainNavNode::routeGuidanceObjective(short_route,length_cost),
              TerrainNavNode::routeGuidanceObjective(detour,length_cost));
    EXPECT_GT(TerrainNavNode::routeGuidanceObjective(short_route,risk_cost),
              TerrainNavNode::routeGuidanceObjective(detour,risk_cost));
    EXPECT_FALSE(std::isfinite(TerrainNavNode::routeGuidanceObjective(short_route,
      [](double,double,double){return double(NAN);}))); 
    short_route.reached=false;
    EXPECT_FALSE(std::isfinite(TerrainNavNode::routeGuidanceObjective(short_route,length_cost)));
  }
  static void tracking_reserve_prefers_clearance_without_turning_it_into_a_hard_margin()
  {
    auto n=node(1e-8F);n->body_collision_front_=.58;n->body_collision_back_=.48;
    n->body_collision_half_width_=.40;n->footprint_margin_=.10;
    const auto body=n->collisionEnvelope(.1);
    const std::vector<std::array<double,4>> rectangles{{.4,.53,1.8,.54}};
    EXPECT_FALSE(bodyOverlapsRectangle(0.,0.,0.,body.front,body.back,body.half_width,rectangles[0]));
    EXPECT_EQ(TerrainNavNode::trackingReservePenalty(rectangles,body,.06,20.,0.,0.,0.),20.);
    EXPECT_EQ(TerrainNavNode::trackingReservePenalty(rectangles,body,.06,20.,0.,-.2,0.),0.);
    terrain_nav::RouteMemory::Result tight,wide;tight.reached=wide.reached=true;
    tight.path={{0,0},{1,0},{2,0},{3,0}};wide.path={{0,0},{1,-.2},{2,-.2},{3,0}};
    tight.yaws=wide.yaws=std::vector<double>(4,0.);
    const auto cost=[&](double x,double y,double yaw){
      return 1.+TerrainNavNode::trackingReservePenalty(rectangles,body,.06,20.,x,y,yaw);
    };
    EXPECT_LT(TerrainNavNode::routeGuidanceObjective(wide,cost),TerrainNavNode::routeGuidanceObjective(tight,cost));
    EXPECT_NEAR(body.front,.68,1e-12);EXPECT_NEAR(body.back,.58,1e-12);EXPECT_DOUBLE_EQ(body.half_width,.5);
  }
  static void unavailable_preferred_reserve_does_not_block_the_nominal_narrow_route()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
    n->background_guide_runtime_=true;n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;n->footprint_margin_=.1;
    n->reference_tracking_reserve_m_=.06;n->map_stamp_=n->now();
    n->route_memory_.observe(1.,{-4,-4,4,4},{},{},{},{},false,{{.2,.54,.005}});
    OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
    terrain_nav::RouteMemory::Result foreground;
    ASSERT_TRUE(n->completeForwardSeedLocked(odom,goal,foreground));
    std::vector<float> lower(n->width_*n->height_,1.);terrain_nav::RouteOptions options;
    options.heuristic_weight=2.5;options.advisory_cost=2.;
    n->queueBackgroundGuideLocked(odom,goal,lower,options);
    ASSERT_TRUE(n->background_guide_.valid());const auto result=n->background_guide_.get();
    ASSERT_TRUE(result.route.reached);EXPECT_STREQ(result.candidate,"risk_proposal");
    EXPECT_FALSE(result.geometry_reached);EXPECT_TRUE(result.risk_reached);
    EXPECT_FALSE(n->have_path_);EXPECT_DOUBLE_EQ(n->footprint_margin_,.1);
  }
  static void observation_background_never_changes_history_or_authorizes_unknown_execution()
  {
    for(int fault=0;fault<3;++fault) {
      auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
      n->background_guide_runtime_=true;n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
      n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.40;
      n->footprint_margin_=.10;
      n->map_stamp_=n->now();OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
      n->route_memory_.observe(1.,{-4,-4,4,4},{},{{0.,.4}});
      terrain_nav::RouteMemory current(n->route_memory_.resolution());
      current.observe(2.,{-4,-4,4,4},{},{});
      terrain_nav::RouteOptions options;options.advisory_cost=2.;
      std::vector<float> lower(n->width_*n->height_,1.);
      n->queueBackgroundGuideLocked(odom,goal,lower,options,{},&current);
      ASSERT_TRUE(n->background_guide_.valid());current.clear();
      auto result=n->background_guide_.get();ASSERT_TRUE(result.route.reached);
      EXPECT_STREQ(result.candidate,"current_evidence_observation");
      EXPECT_FALSE(result.geometry_reached);EXPECT_FALSE(result.risk_reached);
      EXPECT_TRUE(n->route_memory_.occupiedAt({0.,.4}));
      if(fault==1)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      if(fault==2)for(auto &slice:n->map_["collision_risk"])slice.setOnes();
      const auto sequence=n->path_seq_;
      std::promise<TerrainNavNode::BackgroundGuideResult> ready;n->background_guide_=ready.get_future();
      ready.set_value(std::move(result));terrain_nav::RouteMemory::Result selected;
      EXPECT_EQ(n->takeBackgroundGuideLocked(odom,goal,selected),fault==0);
      EXPECT_EQ(n->path_seq_,sequence);EXPECT_FALSE(n->have_path_);
      EXPECT_DOUBLE_EQ(n->footprint_margin_,.10);EXPECT_DOUBLE_EQ(n->chance_path_budget_,.10);
    }
  }
  static void background_maneuver_keeps_future_gears_but_executes_only_a_fresh_stopped_prefix()
  {
    for(int fault=0;fault<8;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
      n->rear_axle_offset_=.2078;OdomState odom;odom.valid=true;
      terrain_nav::RouteMemory::Result route;route.reached=true;
      auto p=Pose2{};
      for(int i=0;i<=10;++i) {
        p=n->advanceArc(Pose2{},0.,-.1*i);
        route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(-1);
      }
      const auto cusp=p;
      for(int i=1;i<=25;++i) {
        p=n->advanceArc(cusp,.4,.1*i);
        route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(1);
      }
      Pose2 goal{p.x,p.y,p.yaw};
      TerrainNavNode::BackgroundGuideResult proposal;proposal.route=route;proposal.goal=goal;
      proposal.source_stamp=n->now().seconds();proposal.spatial=n->spatial_epoch_.load();
      if(fault==1)proposal.source_stamp-=3.;
      if(fault==2)++proposal.spatial;
      if(fault==3)proposal.goal.x+=1.;
      if(fault==4)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      if(fault==5)odom.v=.1;
      if(fault==6)odom.y=.2;
      if(fault==7) {
        for(auto &slice:n->map_["probability_valid"])for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
          Eigen::Vector3d q;n->map_.index2Pos(Eigen::Array3i(r,c,0),q);if(q.x()<-.55)slice(r,c)=0.;
        }
      }
      n->background_maneuver_=std::move(proposal);
      std::vector<Pose2> prefix;std::deque<std::vector<Pose2>> future;
      const auto sequence=n->path_seq_;
      const bool admitted=n->takeBackgroundManeuverLocked(odom,goal,prefix,future);
      EXPECT_EQ(admitted,fault==0||fault==7)<<fault;
      EXPECT_EQ(n->path_seq_,sequence);EXPECT_FALSE(n->have_path_);
      if(admitted) {
        ASSERT_FALSE(future.empty());EXPECT_EQ(prefix.front().direction,-1);
        EXPECT_EQ(future.back().front().direction,1);
        EXPECT_NEAR(future.back().back().x,goal.x,1e-8);EXPECT_NEAR(future.back().back().y,goal.y,1e-8);
        EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));EXPECT_TRUE(n->bodyArcPathFeasible(prefix));
        if(fault==7) {
          EXPECT_LT(prefix.back().s,1.);EXPECT_EQ(future.front().front().direction,-1);
          EXPECT_FALSE(n->chancePathSafe(future.front(),nullptr,.1)); // unknown tail stays queued
        }
      }
    }
  }
  static void ready_reverse_cannot_be_starved_by_a_successful_stopping_prefix()
  {
    for(int fault=0;fault<7;++fault) {
      auto n=staged_fixture();n->recovery_path_=false;n->recovery_staged_gears_=false;
      n->recovery_legs_.clear();n->path_.clear();n->path_seq_=11;
      const int count=fault==2?40:3;
      for(int i=0;i<=count;++i)n->path_.push_back({.05*i,0.,0.,.05*i,1});
      n->guide_handoff_=fault==4?TerrainNavNode::GuideHandoff::Computing:TerrainNavNode::GuideHandoff::ReverseReady;
      auto live=std::make_shared<LiveObstacleSnapshot>(*std::atomic_load(&n->live_snapshot_));
      live->hard[25*40+20]=1;live->present_hard=live->hard; // nearby side object, not reverse corridor
      if(fault==6)live->stamp_sec-=10.;
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      terrain_nav::RouteMemory::Result route;route.reached=true;Pose2 p{};
      for(int i=0;i<=10;++i) {
        p=n->advanceArc(Pose2{},0.,-.1*i);
        route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(-1);
      }
      const auto cusp=p;
      for(int i=1;i<=25;++i) {
        p=n->advanceArc(cusp,.4,.1*i);
        route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(1);
      }
      n->raw_goal_={p.x,p.y,p.yaw};
      TerrainNavNode::BackgroundGuideResult proposal;proposal.route=route;proposal.goal=n->raw_goal_;
      proposal.source_stamp=n->now().seconds();proposal.spatial=n->spatial_epoch_.load();
      n->background_maneuver_=std::move(proposal);
      if(fault==1)n->odom_.v=.1;
      if(fault==3)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      const auto epoch=n->input_epoch_.load();bool ok=true,tried=false;
      n->dispatchReadyReverseHandoff(fault==5?2:1,epoch,ok,tried);
      EXPECT_TRUE(ok)<<fault; // failure cannot revoke the old prefix identity
      EXPECT_EQ(tried,fault!=4)<<fault;
      if(fault==0) {
        EXPECT_TRUE(n->recovery_path_);EXPECT_TRUE(n->recovery_staged_gears_);
        EXPECT_GT(n->path_seq_,11u);EXPECT_EQ(n->path_.front().direction,-1);
        EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
      } else {
        EXPECT_FALSE(n->recovery_path_)<<fault;EXPECT_EQ(n->path_seq_,11u)<<fault;
        EXPECT_EQ(n->path_.front().direction,1)<<fault;
      }
    }
  }

  static void background_search_can_find_a_complete_reverse_first_maneuver()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
    n->background_guide_runtime_=true;n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;n->reverse_max_distance_=4.;
    n->map_stamp_=n->now();OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
    terrain_nav::RouteMemory current(n->route_memory_.resolution());
    current.observe(1.,{-4,-4,4,4},{},{},{},{},false,{{.8,0.,.025,.8}});
    n->route_memory_=current;
    std::vector<float> lower(n->width_*n->height_,1.);terrain_nav::RouteOptions options;
    options.heuristic_weight=2.5;options.advisory_cost=2.;
    n->queueBackgroundGuideLocked(odom,goal,lower,options,{},&current);
    ASSERT_TRUE(n->background_guide_.valid());
    const auto result=n->background_guide_.get();
    ASSERT_TRUE(result.route.reached)<<result.route.reason<<" expanded="<<result.route.expanded;
    EXPECT_STREQ(result.candidate,"current_evidence_maneuver");
    ASSERT_FALSE(result.route.directions.empty());EXPECT_EQ(result.route.directions.front(),-1);
    EXPECT_NE(std::find(result.route.directions.begin(),result.route.directions.end(),1),result.route.directions.end());
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
  }

  static void completed_reverse_guide_is_dispatched_by_the_actual_planner_job()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
    n->topology_guide_enable_=true;n->background_guide_runtime_=true;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->hard_clearance_floor_=0.;n->map_stamp_=n->now();n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->goal_seq_=7;n->raw_goal_={3.,0.,0.,0.};
    terrain_nav::RouteMemory current(n->route_memory_.resolution());
    current.observe(1.,{-4,-4,4,4},{},{},{},{},false,{{.8,0.,.025,.8}});n->route_memory_=current;
    std::vector<float> lower(n->width_*n->height_,1.);terrain_nav::RouteOptions options;
    options.heuristic_weight=2.5;options.advisory_cost=2.;
    n->queueBackgroundGuideLocked(n->odom_,n->raw_goal_,lower,options,{},&current);
    ASSERT_TRUE(n->background_guide_.valid());auto proposal=n->background_guide_.get();
    ASSERT_TRUE(proposal.route.reached);ASSERT_EQ(proposal.route.directions.front(),-1);
    // Install the actual blocked-front evidence; it cannot be bypassed by a
    // direct forward connector. Only the first stopped reverse prefix may run.
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4.;live->resolution=.2;
    live->width=live->height=40;live->inflation_radius=0.;live->hard.assign(1600,0);live->stamp_sec=n->now().seconds();
    for(int row=16;row<=23;++row)live->hard[row*40+24]=1;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    std::promise<TerrainNavNode::BackgroundGuideResult> ready;n->background_guide_=ready.get_future();
    ready.set_value(std::move(proposal));n->background_guide_runtime_=false;
    n->odom_.stamp=n->now();n->map_stamp_=n->now();n->max_odom_age_sec_=2.;
    n->reverse_recovery_count_=2; // actual stopped failure: local retries are exhausted
    ASSERT_TRUE(n->executePlannerJob("qualified completed reverse handoff"));
    ASSERT_TRUE(n->have_path_);ASSERT_FALSE(n->path_.empty());
    EXPECT_EQ(n->path_.front().direction,-1);EXPECT_TRUE(n->recovery_staged_gears_);
    EXPECT_GT(n->recovery_legs_.size(),0u);EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));EXPECT_EQ(n->reverse_recovery_total_,1);
    EXPECT_EQ(n->reverse_recovery_count_,2); // never reset the exhausted local guard to pass
  }

  static void complete_forward_geometry_without_supported_entrance_cannot_starve_reverse_search()
  {
    auto n=node(1e-8F);n->topology_forward_seed_enable_=true;n->topology_async_guide_enable_=true;
    n->background_guide_runtime_=true;n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;n->map_stamp_=n->now();
    for(auto &slice:n->map_["probability_valid"])for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
      Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
      if(p.x()>.05 && p.x()<.5 && std::abs(p.y())<.5)slice(r,c)=0.;
    }
    OdomState odom;odom.valid=true;const Pose2 goal{3.,0.,0.};
    terrain_nav::RouteMemory current(n->route_memory_.resolution());current.observe(1.,{-4,-4,4,4},{},{});
    n->route_memory_=current;std::vector<float> lower(n->width_*n->height_,1.);terrain_nav::RouteOptions options;
    options.heuristic_weight=2.5;options.advisory_cost=2.;
    n->queueBackgroundGuideLocked(odom,goal,lower,options,{},&current);
    ASSERT_TRUE(n->background_guide_.valid());auto result=n->background_guide_.get();
    ASSERT_TRUE(result.route.reached)<<result.route.reason;
    EXPECT_STREQ(result.candidate,"supported_reverse_entrance");EXPECT_EQ(result.route.directions.front(),-1);
    EXPECT_TRUE(TerrainNavNode::routeEntranceSupported(result.route,n->map_,.05));
    // This is a proposal only. The measured ground is unchanged, and the
    // unsupported forward states remain forbidden to the actual executor.
    EXPECT_FALSE(n->queryCostLocked(.1,0.,0.).known);EXPECT_FALSE(n->have_path_);
    EXPECT_DOUBLE_EQ(n->chance_state_limit_,.05);EXPECT_DOUBLE_EQ(n->chance_path_budget_,.1);
  }
  static void whole_goal_search_pause_is_bounded_and_scoped_to_progress_and_goal()
  {
    auto n=node(1e-8F);OdomState odom;const Pose2 goal{3.,0.,0.};
    EXPECT_FALSE(n->waitForCompleteGuideLocked(odom,goal));
    std::promise<TerrainNavNode::BackgroundGuideResult> promise;n->background_guide_=promise.get_future();
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    EXPECT_TRUE(n->waitForCompleteGuideLocked(odom,goal));
    ASSERT_EQ(rcl_set_ros_time_override(clock,100500000000LL),RCL_RET_OK);
    EXPECT_TRUE(n->waitForCompleteGuideLocked(odom,goal));
    ASSERT_EQ(rcl_set_ros_time_override(clock,100900000000LL),RCL_RET_OK);
    for(int i=0;i<10;++i)EXPECT_FALSE(n->waitForCompleteGuideLocked(odom,goal));
    EXPECT_TRUE(n->waitForCompleteGuideLocked(odom,Pose2{4.,0.,0.}));
    ASSERT_EQ(rcl_set_ros_time_override(clock,102000000000LL),RCL_RET_OK);
    EXPECT_FALSE(n->waitForCompleteGuideLocked(odom,Pose2{4.,0.,0.}));
    odom.x=.61;EXPECT_TRUE(n->waitForCompleteGuideLocked(odom,Pose2{4.,0.,0.}));
    EXPECT_FALSE(n->have_path_);EXPECT_EQ(n->path_seq_,0u);
  }
  static void far_invalid_suffix_does_not_erase_the_current_certified_turn()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;n->rear_axle_offset_=.2078;
    OdomState odom;odom.valid=true;
    std::vector<Pose2> old;
    for(int i=0;i<=30;++i){auto p=n->advanceArc(Pose2{},.2,.1*i);p.s=.1*i;old.push_back(p);}
    for(auto &slice:n->map_["probability_valid"])for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c) {
      Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()>1.8)slice(r,c)=0.;
    }
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,old,false));
    std::vector<Pose2> prefix;ASSERT_TRUE(n->certifiedRemainingForwardPrefixLocked(odom,old,prefix));
    EXPECT_GE(prefix.back().s,1.2);EXPECT_LT(prefix.back().s,old.back().s);
    EXPECT_GT(prefix.back().y,.2); // preserve the already planned TURN, not a straight ray
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));EXPECT_FALSE(n->chancePathSafe(old,nullptr,.1));
    Pose2 active;std::string reason;std::vector<Pose2> guide;
    ASSERT_TRUE(n->resolveActiveGoalLocked(odom,Pose2{6.,0.,0.},active,reason,guide,old,true));
    EXPECT_EQ(reason,"retain certified current prefix");EXPECT_NEAR(active.y,prefix.back().y,1e-10);
    EXPECT_TRUE(n->bodyArcPathFeasible(guide));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->certifiedRemainingForwardPrefixLocked(odom,old,prefix)); // no certificate, no retention
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["collision_risk"])slice.setOnes();
    EXPECT_FALSE(n->certifiedRemainingForwardPrefixLocked(odom,old,prefix)); // current geometry still wins
  }
  static std::unique_ptr<TerrainNavNode> staged_fixture(double offset=0.)
  {
    auto n=node(1e-8F);n->rear_axle_offset_=offset;n->hard_clearance_floor_=0.;n->live_obstacle_enable_=true;
    n->reverse_enable_=true;n->reverse_recovery_enable_=true;n->odom_.valid=true;n->odom_.stamp=n->now();
    n->map_stamp_=n->now();n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,1,0,0};
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=-4;live->min_y=-4;
    live->resolution=.2;live->width=40;live->height=40;live->stamp_sec=n->now().seconds();
    live->hard.assign(1600,0);live->present_hard=live->hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    terrain_nav::RouteMemory::Result route;Pose2 p{};
    const auto append=[&]{route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(p.direction);};
    append();for(int i=0;i<8;++i){p=n->advanceArc(p,.1,.1);append();}
    for(int i=0;i<6;++i){p=n->advanceArc(p,-.2,-.1);p.direction=-1;append();}
    for(int i=0;i<12;++i){p=n->advanceArc(p,.1,.1);p.direction=1;append();}
    auto legs=n->splitRouteGearLegs(route);
    if(legs.size()!=3)throw std::runtime_error("staged fixture has invalid car geometry");
    n->path_=legs.front();n->have_path_=true;n->recovery_path_=true;n->recovery_staged_gears_=true;
    n->recovery_legs_={legs[1],legs[2]};n->recovery_legs_spatial_epoch_=n->spatial_epoch_.load();
    return n;
  }
  static void recovery_evidence_wait_preserves_intent_without_permission_or_extra_attempts()
  {
    auto n=staged_fixture();const auto end=n->path_.back();
    n->odom_.x=end.x;n->odom_.y=end.y;n->odom_.yaw=end.yaw;n->odom_.v=0.;n->odom_.stamp=n->now();
    n->recovery_geometry_epoch_=n->input_epoch_.load();
    auto live=std::make_shared<LiveObstacleSnapshot>(*std::atomic_load(&n->live_snapshot_));
    // The stopped body is clear; a newly occupied cell invalidates the NEXT
    // reverse leg before it has enough certified length to execute.
    const int col=int(std::floor((.1-live->min_x)/live->resolution));
    const int row=int(std::floor((.1-live->min_y)/live->resolution));
    live->hard[row*live->width+col]=1;live->present_hard=live->hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto count=n->reverse_recovery_total_.load();const auto sequence=n->path_seq_;
    const auto legs=n->recovery_legs_.size();
    ASSERT_FALSE(n->planPath("new obstacle at stopped gear cusp"));
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->recovery_resume_pending_);
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);EXPECT_EQ(n->recovery_legs_.size(),legs);
    EXPECT_EQ(n->path_seq_,sequence);EXPECT_EQ(n->reverse_recovery_total_.load(),count);
    EXPECT_FALSE(n->planReverseRecovery(n->goal_seq_,n->input_epoch_.load()));
    EXPECT_EQ(n->reverse_recovery_total_.load(),count);
    live=std::make_shared<LiveObstacleSnapshot>(*live);std::fill(live->hard.begin(),live->hard.end(),0);
    live->present_hard=live->hard;live->stamp_sec=n->now().seconds();
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    ASSERT_TRUE(n->planPath("fresh evidence clears next gear"));
    EXPECT_TRUE(n->have_path_);EXPECT_FALSE(n->recovery_resume_pending_);
    EXPECT_EQ(n->path_.front().direction,-1);
    EXPECT_EQ(n->reverse_recovery_total_.load(),count+1);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
  }

  static void recovery_wait_deadline_is_not_renewed_and_cannot_cross_identity()
  {
    auto n=staged_fixture();const auto epoch=n->input_epoch_.load(),sequence=n->path_seq_;
    n->recovery_geometry_epoch_=epoch;
    ASSERT_TRUE(n->pauseRecoveryForEvidence(n->goal_seq_,epoch,sequence));
    n->recovery_resume_saved_time_-=4.;const auto start=n->recovery_resume_saved_time_;
    ASSERT_TRUE(n->pauseRecoveryForEvidence(n->goal_seq_,epoch,sequence));
    EXPECT_DOUBLE_EQ(n->recovery_resume_saved_time_,start);
    EXPECT_FALSE(n->pauseRecoveryForEvidence(n->goal_seq_+1,epoch,sequence));
    EXPECT_FALSE(n->pauseRecoveryForEvidence(n->goal_seq_,epoch+1,sequence));
    EXPECT_FALSE(n->pauseRecoveryForEvidence(n->goal_seq_,epoch,sequence+1));
    n->recovery_resume_saved_time_-=2.;
    EXPECT_FALSE(n->pauseRecoveryForEvidence(n->goal_seq_,epoch,sequence));
    EXPECT_FALSE(n->have_path_);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
  }

  static void staged_gears_stop_and_recertify_every_direction_change()
  {
    auto n=staged_fixture();
    const auto at_end=[&](double speed){const auto p=n->path_.back();n->odom_.x=p.x;n->odom_.y=p.y;n->odom_.yaw=p.yaw;n->odom_.v=speed;n->odom_.stamp=n->now();};
    at_end(.1);ASSERT_TRUE(n->planPath("forward cusp still moving"));EXPECT_EQ(n->path_.front().direction,1);
    at_end(0.);ASSERT_TRUE(n->planPath("forward cusp stopped"));EXPECT_EQ(n->path_.front().direction,-1);
    EXPECT_EQ(n->recovery_legs_.size(),1u);EXPECT_EQ(n->reverse_recovery_total_,1);EXPECT_EQ(n->reverse_recovery_count_,0);
    EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    at_end(-.1);ASSERT_TRUE(n->planPath("reverse cusp still moving"));EXPECT_EQ(n->path_.front().direction,-1);
    at_end(0.);ASSERT_TRUE(n->planPath("reverse cusp stopped"));EXPECT_EQ(n->path_.front().direction,1);
    EXPECT_TRUE(n->recovery_legs_.empty());EXPECT_EQ(n->reverse_recovery_total_,1);
    EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();goal->pose.position.x=2.;goal->pose.orientation.w=1.;
    n->goalCallback(goal);EXPECT_FALSE(n->recovery_staged_gears_);EXPECT_TRUE(n->recovery_legs_.empty());
  }
  static void staged_gears_cannot_dispatch_unknown_stale_colliding_or_over_budget_legs()
  {
    for(int failure=0;failure<5;++failure) {
      auto n=staged_fixture();auto end=n->path_.back();n->odom_.x=end.x;n->odom_.y=end.y;n->odom_.yaw=end.yaw;n->odom_.v=0.;
      // Unknown starts within the first minimum executable prefix. A distant
      // unknown tail is covered separately by the rolling-prefix test.
      if(failure==0)for(auto &slice:n->map_["probability_valid"])
        for(int r=0;r<slice.rows();++r)for(int c=0;c<slice.cols();++c){Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);if(p.x()<.65)slice(r,c)=0.;}
      if(failure==0)ASSERT_TRUE(n->queryCostLocked(end.x,end.y,end.yaw).known);
      if(failure==1) {
        auto live=std::make_shared<LiveObstacleSnapshot>(*std::atomic_load(&n->live_snapshot_));live->hard[20*40+20]=1;
        std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      }
      if(failure==2)n->map_stamp_=n->now()-rclcpp::Duration::from_seconds(2.);
      if(failure==3)n->spatial_epoch_.fetch_add(1);
      if(failure==4)n->reverse_recovery_total_=16;
      EXPECT_FALSE(n->planPath("staged next leg changed"));
      EXPECT_EQ(n->last_cmd_v_,0.);EXPECT_EQ(n->reverse_recovery_total_,failure==4?16:0);
      EXPECT_FALSE(n->have_path_);
      if(failure<3) {
        // Missing evidence, occupancy and age revoke execution immediately;
        // coordinate-only intentions may survive the bounded STOPPED wait.
        EXPECT_TRUE(n->recovery_staged_gears_);EXPECT_TRUE(n->recovery_resume_pending_);
        EXPECT_FALSE(n->recovery_legs_.empty());
        n->recovery_resume_saved_time_-=6.;
        n->planPath("evidence wait expired");
        EXPECT_FALSE(n->recovery_staged_gears_);EXPECT_FALSE(n->recovery_resume_pending_);
        EXPECT_TRUE(n->recovery_legs_.empty());EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
        // Expiring an old reverse intent does not prohibit a NEW ordinary
        // forward solution. It still needs its own current certificate, and
        // actual control independently rejects stale source ages.
        if(n->have_path_) {
          EXPECT_FALSE(n->recovery_path_);EXPECT_EQ(n->path_.front().direction,1);
          EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
        }
      } else {
        // A changed coordinate epoch or exhausted transition budget cannot
        // become a temporary evidence wait and must discard the old intent.
        EXPECT_FALSE(n->recovery_staged_gears_);EXPECT_FALSE(n->recovery_resume_pending_);
      }
    }
  }
  static void staged_local_join_preserves_world_fixed_future_gates_and_goal()
  {
    auto n=staged_fixture();const auto reverse=n->recovery_legs_.front(),forward=n->recovery_legs_.back();
    OdomState actual;actual.x=reverse.front().x+.04;actual.y=reverse.front().y-.02;actual.yaw=reverse.front().yaw+.03;
    const auto r=n->joinFixedGearLegLocked(reverse,actual),f=forward;
    ASSERT_FALSE(r.empty());ASSERT_FALSE(f.empty());EXPECT_EQ(r.front().direction,-1);EXPECT_EQ(f.front().direction,1);
    EXPECT_NEAR(r.front().x,actual.x,1e-10);EXPECT_NEAR(r.front().yaw,actual.yaw,1e-10);
    EXPECT_NEAR(r.back().x,f.front().x,1e-10);EXPECT_NEAR(r.back().y,f.front().y,1e-10);
    EXPECT_NEAR(r.back().yaw,reverse.back().yaw,1e-10);
    n->odom_=actual;n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=0.;
    ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    ASSERT_EQ(n->recovery_legs_.size(),1u);
    const auto &fixed=n->recovery_legs_.front();ASSERT_EQ(fixed.size(),forward.size());
    for(size_t i=0;i<fixed.size();++i) {
      EXPECT_DOUBLE_EQ(fixed[i].x,forward[i].x);EXPECT_DOUBLE_EQ(fixed[i].y,forward[i].y);
      EXPECT_DOUBLE_EQ(fixed[i].yaw,forward[i].yaw);
    }
    EXPECT_TRUE(n->bodyArcPathFeasible(r));EXPECT_TRUE(n->bodyArcPathFeasible(f));
  }
  static void global_proposal_selection_uses_the_full_frontier_after_certifying_its_local_prefix()
  {
    auto n=node(1e-8F);n->local_goal_distance_=2.;
    terrain_nav::RouteMemory::Result straight,through;straight.partial=through.partial=true;
    const auto append=[](auto &route,const auto &p){route.path.push_back({p.x,p.y});route.yaws.push_back(p.yaw);route.directions.push_back(1);};
    Pose2 a{},b{};append(straight,a);append(through,b);
    for(int i=0;i<30;++i){a=advanceArc(a,0.,.1);append(straight,a);}
    for(int i=0;i<80;++i){b=advanceArc(b,i<20?.3:0.,.1);append(through,b);}
    const Pose2 goal{10,0,0,0};
    // The through proposal makes LESS immediate x progress in its first 2m,
    // but its full frontier advances beyond the straight dead-end entrance.
    EXPECT_LT(through.path[20][0],straight.path[20][0]);
    const auto short_score=n->certifiedTopologyProposalScoreLocked(straight,goal);
    const auto through_score=n->certifiedTopologyProposalScoreLocked(through,goal);
    EXPECT_TRUE(std::isfinite(through_score));EXPECT_LT(through_score,short_score);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(std::isfinite(n->certifiedTopologyProposalScoreLocked(through,goal)));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    through.directions.back()=-1;
    EXPECT_FALSE(std::isfinite(n->certifiedTopologyProposalScoreLocked(through,goal))); // hidden future gears cannot become forward execution
  }
  static void failed_branch_preference_is_refreshed_at_the_stopped_cusp_without_changing_risk()
  {
    auto n=active_recovery();
    auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(TerrainNavNode::RecoveryProgressOrigin{0,0,n->goal_seq_,0});
    std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
    n->rememberBlockedRoute(n->odom_,n->goal_seq_);
    auto memory=std::make_shared<std::vector<TerrainNavNode::RecoveryRouteMemory>>(*std::atomic_load(&n->recovery_route_memory_));
    memory->front().stamp=n->now().seconds()-14.8;
    std::atomic_store(&n->recovery_route_memory_,std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>(memory));
    EXPECT_LT(n->recoveryRoutePenalty(0,0),.3);
    const double p=n->queryCostLocked(0,0,0).probability_upper;
    const auto end=n->path_.back();n->odom_.x=end.x;n->odom_.y=end.y;n->odom_.yaw=end.yaw;n->odom_.v=0.;
    ASSERT_TRUE(n->planPath("confirmed stopped recovery cusp"));
    EXPECT_GT(n->recoveryRoutePenalty(0,0),14.);
    EXPECT_EQ(n->queryCostLocked(0,0,0).probability_upper,p);
    EXPECT_EQ(n->reverse_recovery_count_,1);EXPECT_EQ(n->path_.front().direction,1);
  }
  static void routing_geometry_survives_temporal_pause_but_not_a_spatial_generation_change()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->map_stamp_=n->now();
    n->route_memory_.observe(1.,{8,-1,12,1},{},{{10.,0.}});
    n->route_memory_epoch_=n->spatial_epoch_.load();
    EXPECT_TRUE(n->route_memory_.occupiedAt({10,0}));
    n->waitForFreshInputs();n->stale_inputs_waiting_.store(false);
    OdomState odom;Pose2 active;std::vector<Pose2> guide;
    n->resolveTopologyGoalLocked(odom,Pose2{3,0,0,0},active,guide,true);
    EXPECT_TRUE(n->route_memory_.occupiedAt({10,0}));
    n->spatial_epoch_.fetch_add(1);
    n->resolveTopologyGoalLocked(odom,Pose2{3,0,0,0},active,guide,true);
    EXPECT_FALSE(n->route_memory_.occupiedAt({10,0}));
  }
  static void temporal_pauses_preserve_measured_forward_progress_but_frame_changes_do_not_release_retries()
  {
    for(bool frame_fault:{false,true}) {
      auto n=node(1e-8F);n->topology_guide_enable_=true;n->have_goal_=true;n->goal_seq_=1;
      n->raw_goal_={3,0,0,0};n->max_odom_age_sec_=2.;n->reverse_recovery_count_=2;n->reverse_recovery_total_=7;
      auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(TerrainNavNode::RecoveryProgressOrigin{0,0,1,0});
      std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
      n->recovery_progress_valid_epoch_.store(0);
      const auto start=n->now()-rclcpp::Duration::from_seconds(1.);
      OdomState odom;odom.valid=true;odom.v=.5;
      for(int k=0;k<=8;++k) {
        if(k==3||k==6) {
          n->waitForFreshInputs();EXPECT_EQ(n->last_cmd_v_,0.);
          n->stale_inputs_waiting_.store(false);
          EXPECT_EQ(std::atomic_load(&n->reverse_progress_origin_).get(),origin.get());
        }
        n->last_cmd_v_=.5;odom.x=.1*k;odom.stamp=start+rclcpp::Duration::from_seconds(.1*k);
        n->recordCertifiedRecoveryForwardProgress(odom,true);
      }
      const auto progress=std::atomic_load(&n->recovery_forward_progress_);
      ASSERT_TRUE(progress);EXPECT_NEAR(progress->distance,.8,1e-9);
      if(frame_fault)n->input_epoch_.fetch_add(1); // not a temporal wait/rebase
      n->completeTopologyRecovery(1,n->input_epoch_.load(),odom,origin);
      EXPECT_EQ(n->reverse_recovery_count_,frame_fault?2:0);
      EXPECT_EQ(n->reverse_recovery_total_,7);
    }
  }
  static void paused_recovery_requires_fresh_recertification_and_does_not_consume_another_attempt()
  {
    for(int fault=0;fault<5;++fault) {
      auto n=active_recovery();n->map_stamp_=n->now();
      auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=-4;live->min_y=-4;
      live->resolution=.2;live->width=40;live->height=40;live->stamp_sec=n->now().seconds();
      live->hard.assign(1600,0);live->present_hard=live->hard;
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      const auto geometry=n->path_;
      n->waitForFreshInputs();
      EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->recovery_resume_pending_);EXPECT_EQ(n->last_cmd_v_,0.);
      EXPECT_EQ(n->path_.size(),geometry.size());EXPECT_EQ(n->reverse_recovery_count_,1);
      n->stale_inputs_waiting_.store(false); // emulate the completed fresh-interval gate
      if(fault==1)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      if(fault==2)live->hard.assign(1600,1);
      if(fault==3)n->input_epoch_.fetch_add(1); // incompatible map/frame generation
      if(fault==4)n->recovery_resume_saved_time_-=6.;
      n->planPath("fresh inputs restored");
      if(!fault) {
        ASSERT_TRUE(n->have_path_);EXPECT_FALSE(n->recovery_resume_pending_);
        EXPECT_TRUE(n->recovery_path_);EXPECT_EQ(n->path_.front().direction,-1);
      } else {
        // A separate new forward solution may exist for epoch/expiry faults;
        // the suspended reverse geometry must never be re-authorized.
        EXPECT_FALSE(n->have_path_ && n->recovery_path_);
      }
      EXPECT_EQ(n->reverse_recovery_count_,1);
    }
  }
  static void deadend_retreat_is_finite_and_requires_current_whole_body_ground_and_budget()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->reverse_enable_=true;n->hard_clearance_floor_=0.;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
    n->footprint_margin_=.1;n->map_stamp_=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=-4;live->min_y=-4;live->resolution=.2;live->width=40;live->height=40;
    live->stamp_sec=n->now().seconds();live->hard.assign(1600,0);live->present_hard=live->hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    std::vector<Pose2> retreat;
    ASSERT_TRUE(n->buildCertifiedDeadendRetreatLocked(Pose2{0,0,0,0,-1},retreat));
    EXPECT_NEAR(retreat.back().s,2.5,1e-6);EXPECT_NEAR(retreat.back().x,-2.5,1e-6);
    EXPECT_TRUE(n->bodyArcPathFeasible(retreat));EXPECT_TRUE(n->chancePathSafe(retreat,nullptr,.1));
    for(const auto &p:retreat)EXPECT_EQ(p.direction,-1);
    live->stamp_sec-=2.;EXPECT_FALSE(n->buildCertifiedDeadendRetreatLocked(Pose2{},retreat));
    live->stamp_sec=n->now().seconds();
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->buildCertifiedDeadendRetreatLocked(Pose2{},retreat));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(.04F);
    EXPECT_FALSE(n->buildCertifiedDeadendRetreatLocked(Pose2{},retreat));
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(1e-8F);
    for(int y=0;y<40;++y)live->hard[y*40+16]=1; // rear wall [-.8,-.6]
    EXPECT_FALSE(n->buildCertifiedDeadendRetreatLocked(Pose2{},retreat));
    live->hard.assign(1600,0);n->map_stamp_=n->now()-rclcpp::Duration::from_seconds(2.);
    EXPECT_FALSE(n->buildCertifiedDeadendRetreatLocked(Pose2{},retreat));
  }
  static void deadend_retreat_stops_before_replanning_without_claiming_a_forward_exit()
  {
    auto n=active_recovery();n->recovery_replan_only_=true;n->recovery_exit_.clear();
    const auto end=n->path_.back();n->odom_.x=end.x;n->odom_.y=end.y;n->odom_.yaw=end.yaw;
    n->odom_.v=-.10;
    ASSERT_TRUE(n->planPath("dead end retreat still moving"));
    EXPECT_TRUE(n->recovery_path_);EXPECT_EQ(n->path_.front().direction,-1);
    n->odom_.v=0.;
    // It can now produce a NEW forward route, but must not resurrect an
    // unproved exit or count reverse displacement as recovery completion.
    n->planPath("dead end retreat stopped");
    EXPECT_FALSE(n->recovery_path_);EXPECT_FALSE(n->recovery_replan_only_);
    if(n->have_path_)EXPECT_EQ(n->path_.front().direction,1);
    EXPECT_TRUE(n->have_goal_);EXPECT_EQ(n->reverse_recovery_count_,1);
    n->recovery_replan_only_=true;
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();goal->pose.position.x=3.;goal->pose.position.y=1.;goal->pose.orientation.w=1.;
    n->goalCallback(goal);EXPECT_FALSE(n->recovery_replan_only_);
  }
  static void control_and_search_use_the_same_protected_gap_and_probability_budget()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
    n->footprint_margin_=.10;n->hard_clearance_floor_=0.;n->clearance_distance_=.65;n->unknown_cost_=60.;
    n->map_.add("collision_subcell_contract",false,3.);
    n->map_.add("collision_subcell_factor",false,2.);
    n->map_.add("collision_subcell_bits",false,0.);
    n->map_.add("terrain_failure_probability_upper",true,1e-8F);
    for(double x=-2.5;x<2.6;x+=.2)for(bool upper:{false,true}) {
      Eigen::Array3i i;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(x,upper?.7:-.5,0),i));
      n->map_.at("collision_risk",i)=1.;n->map_.at("collision_subcell_bits",i)=upper?12.:3.;
    }
    const auto publish=[&] {
      auto s=std::make_shared<ControlMapSnapshot>();s->grid=n->map_;s->stamp=n->now();
      n->buildCostmap(*s);n->installSearchMapLocked(*s);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(s));
    };
    publish();std::vector<Pose2> path;
    for(int i=0;i<=12;++i)path.push_back(Pose2{-.6+.1*i,.12,0,.1*i});
    ASSERT_TRUE(n->pathFootprintSafeLocked(path,.025));
    ASSERT_TRUE(n->chancePathSafe(path,nullptr,.10));
    EXPECT_FALSE(n->pathAheadUnsafe(path,0));
    // Each pose is below 5%, but their accumulated account exceeds 10%.
    for(auto &v:n->map_["terrain_failure_probability_upper"])v.setConstant(.04F);
    publish();EXPECT_TRUE(n->pathAheadUnsafe(path,0));
    for(auto &v:n->map_["terrain_failure_probability_upper"])v.setConstant(1e-8F);
    for(auto &v:n->map_["probability_valid"])v.setZero();
    publish();EXPECT_TRUE(n->pathAheadUnsafe(path,0));
    for(auto &v:n->map_["probability_valid"])v.setOnes();
    n->map_["collision_subcell_bits"][0].setConstant(15.);
    publish();EXPECT_TRUE(n->pathAheadUnsafe(path,0));
  }
  static void a_soft_historical_preference_cannot_close_a_physically_open_corridor()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->historical_obstacle_cost_weight_=24.;n->historical_obstacle_influence_distance_=2.;
    n->historical_clearance_.assign(n->width_*n->height_,.6F);
    ASSERT_GT(n->historicalObstacleCost(1.5,0),.15*24.);
    double blocked=0.;
    EXPECT_FALSE(n->directCorridorBlockedLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},3.,blocked));
    Eigen::Array3i index;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(2,0,0),index));
    n->map_.at("collision_risk",index)=1.;n->rebuildCostmapLocked();
    EXPECT_TRUE(n->directCorridorBlockedLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},3.,blocked));
  }
  static void certified_forward_path_is_revalidated_before_a_different_topology_is_searched()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->topology_guide_enable_=true;
    OdomState odom;odom.valid=true;Pose2 active;std::string reason;std::vector<Pose2> guide;
    const std::vector<Pose2> old={{0,0,0,0},{.5,0,0,.5},{1.,0,0,1.},{1.5,0,0,1.5},{2.,0,0,2.}};
    ASSERT_TRUE(n->resolveActiveGoalLocked(odom,Pose2{6,1.,0,0},active,reason,guide,old,true));
    EXPECT_EQ(reason,"retain certified current path");EXPECT_DOUBLE_EQ(active.x,2.);
    Eigen::Array3i hit;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.,0,0),hit));
    n->map_.at("collision_risk",hit)=1.;
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,old,false));
    n->map_.at("collision_risk",hit)=0.;
    for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,old,false));
  }
  static void local_recovery_can_use_a_useful_frontier_without_claiming_final_arrival()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->reverse_enable_=true;
    n->max_search_time_ms_=200.;n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    Eigen::Array3i index;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(3,0,0),index));
    n->map_.at("collision_risk",index)=1.;n->rebuildCostmapLocked();
    std::vector<Pose2> reverse,exit;
    ASSERT_TRUE(n->buildTopologyRecoveryLocked(Pose2{0,0,0,0},n->raw_goal_,reverse,exit));
    ASSERT_FALSE(reverse.empty());ASSERT_FALSE(exit.empty());
    EXPECT_GE(exit.back().s,1.2);
    EXPECT_GT(std::hypot(exit.back().x,exit.back().y),.6);
    EXPECT_TRUE(n->pathFootprintSafeLocked(reverse,.025));EXPECT_TRUE(n->pathFootprintSafeLocked(exit,.025));
    auto joined=reverse;joined.insert(joined.end(),exit.begin(),exit.end());
    EXPECT_TRUE(n->chancePathSafe(joined,nullptr,.1));
    EXPECT_DOUBLE_EQ(n->raw_goal_.x,3.);EXPECT_TRUE(n->have_goal_);
    double cost=0.;EXPECT_FALSE(n->footprintCostLocked(3,0,0,cost));
    for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->buildTopologyRecoveryLocked(Pose2{0,0,0,0},n->raw_goal_,reverse,exit));
  }
  static void optimistic_exploration_never_prices_unknown_below_a_certified_pose()
  {
    auto n=node(.049F);n->comfort_clearance_weight_=5.2;n->comfort_clearance_distance_=1.35;
    n->comfort_clearance_.assign(n->width_*n->height_,0.);
    const double certified=n->topologyTerrainCostLocked(0,0,0);
    ASSERT_GT(certified,3.);EXPECT_LT(certified,13.2);
    EXPECT_FALSE(n->queryCostLocked(0,0,0).lethal);
    for(auto &v:n->map_["probability_valid"])v.setZero();
    const double unknown=n->topologyTerrainCostLocked(0,0,0);
    EXPECT_DOUBLE_EQ(unknown,13.2);EXPECT_GT(unknown,certified);
    EXPECT_TRUE(std::isfinite(n->topologyTerrainCostLocked(20,20,0)));
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
    double cost=0.;EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    for(auto &v:n->map_["probability_valid"])v.setOnes();
    for(auto &v:n->map_["failure_probability_upper"])v.setOnes();
    EXPECT_GT(n->topologyTerrainCostLocked(0,0,0),unknown);
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
    EXPECT_DOUBLE_EQ(n->chance_state_limit_,.05);EXPECT_DOUBLE_EQ(n->chance_path_budget_,.10);
  }
  static void one_protected_body_passes_a_real_gap_but_cannot_turn_through_its_walls()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
    n->footprint_margin_=.10;
    n->map_.add("collision_subcell_contract",false,3.);
    n->map_.add("collision_subcell_factor",false,2.);
    n->map_.add("collision_subcell_bits",false,0.);
    n->map_.add("terrain_failure_probability_upper",true,1e-8F);
    std::vector<Eigen::Array3i> upper,lower;
    for(double x=-1.5;x<1.6;x+=.2) {
      Eigen::Array3i u,l;
      ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(x,.7,0),u));
      ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(x,-.5,0),l));
      upper.push_back(u);lower.push_back(l);
      for(const auto &i:{u,l}) {
        n->map_.at("collision_risk",i)=1.;n->map_.at("collision_subcell_bits",i)=15.;
      }
    }
    n->rebuildCostmapLocked();double cost=0;
    EXPECT_FALSE(n->footprintCostLocked(0,.12,0,cost));
    // Actual wall faces y=-.5 and .7 leave a 1.2m opening. The 1.0m body
    // ALREADY includes its one 10cm margin; no second dilation is permitted.
    for(const auto &i:upper)n->map_.at("collision_subcell_bits",i)=12.;
    for(const auto &i:lower)n->map_.at("collision_subcell_bits",i)=3.;
    n->rebuildCostmapLocked();
    for(double x=-.6;x<=.6;x+=.1) {
      EXPECT_TRUE(n->footprintCostLocked(x,.12,0,cost))<<x;
      EXPECT_TRUE(n->chancePathSafe({Pose2{x,.12,0,0},Pose2{x+.05,.12,0,.05}},nullptr,.1));
    }
    EXPECT_FALSE(n->footprintCostLocked(0,.12,M_PI/4,cost));
    for(const auto &i:upper)n->map_.at("collision_subcell_bits",i)=15.;
    for(const auto &i:lower)n->map_.at("collision_subcell_bits",i)=15.;
    n->rebuildCostmapLocked();EXPECT_FALSE(n->footprintCostLocked(0,.12,0,cost));
  }
  static void historical_subcells_preserve_whole_square_fallback()
  {
    auto n=node(1e-8F);Eigen::Array3i index;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(0,0,0),index));
    EXPECT_EQ(n->historicalObstacleSubcellBits(n->map_,index),15);
    n->map_.add("historical_obstacle_subcell_bits",false,2.F);
    n->map_.add("historical_obstacle_subcell_factor",false,2.F);
    EXPECT_EQ(n->historicalObstacleSubcellBits(n->map_,index),2);
    for(float value:{0.F,-1.F,16.F,2.5F,std::numeric_limits<float>::quiet_NaN()}) {
      n->map_.at("historical_obstacle_subcell_bits",index)=value;
      EXPECT_EQ(n->historicalObstacleSubcellBits(n->map_,index),15);
    }
    n->map_.at("historical_obstacle_subcell_bits",index)=1.F;
    n->map_.at("historical_obstacle_subcell_factor",index)=4.F;
    EXPECT_EQ(n->historicalObstacleSubcellBits(n->map_,index),15);
  }
  static void finer_obstacle_geometry_does_not_shrink_the_protected_car_or_padding()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;n->footprint_margin_=.1;
    n->map_.add("collision_subcell_contract",false,3.);n->map_.add("collision_subcell_factor",false,2.);
    n->map_.add("collision_subcell_bits",false,0.);n->map_.add("terrain_failure_probability_upper",true,1e-8F);
    Eigen::Array3i index;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.1,.5,0),index));
    n->map_.at("collision_risk",index)=1.;n->map_.at("collision_subcell_bits",index)=12.;
    n->rebuildCostmapLocked();double cost=0.;EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    // Same actual witness box x=[.10,.14], y=[.56,.60] from (.12,.58)
    // plus its unchanged 2cm padding. 10cm raster cells touched y=.50;
    // 5cm cells contain the same witness without inventing that contact.
    n->map_.at("collision_subcell_factor",index)=4.;n->map_.at("collision_subcell_bits",index)=0x6000;
    EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
    EXPECT_FALSE(n->footprintCostLocked(0,.08,0,cost));
    EXPECT_FALSE(n->footprintCostLocked(0,0,M_PI/4,cost));
    for(double bad:{0.,65536.,std::numeric_limits<double>::quiet_NaN()}) {
      n->map_.at("collision_subcell_bits",index)=bad;
      EXPECT_EQ(n->collisionSubcellBits(n->map_,index),65535);
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    }
    n->map_.at("collision_subcell_bits",index)=0x6000;
    for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    EXPECT_DOUBLE_EQ(n->footprint_margin_,.1);
  }
  static void measured_subcell_boxes_do_not_add_unwitnessed_square_extent()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;n->footprint_margin_=.1;
    n->map_.add("collision_subcell_contract",false,3.);n->map_.add("collision_subcell_factor",false,4.);
    n->map_.add("collision_subcell_bits",false,0.);n->map_.add("terrain_failure_probability_upper",true,1e-8F);
    Eigen::Array3i i;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.1,.5,0),i));
    n->map_.at("collision_risk",i)=1.;n->map_.at("collision_subcell_bits",i)=(1<<10)|(1<<14);
    n->rebuildCostmapLocked();double cost=0.;EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    n->map_.add("collision_subcell_xy_bounds_contract",false,1.);
    for(int k=0;k<16;++k)n->map_.add("collision_subcell_xy_bounds_"+std::to_string(k),false,0xF0F0);
    // Original (.12,.535) return, STILL padded by +/- .02m in both axes.
    n->map_.at("collision_subcell_xy_bounds_10",i)=0xF4C0;
    n->map_.at("collision_subcell_xy_bounds_14",i)=0x20C0;
    EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
    EXPECT_FALSE(n->footprintCostLocked(0,.04,0,cost));
    for(double bad:{-1.,65536.,0.,std::numeric_limits<double>::quiet_NaN()}) {
      n->map_.at("collision_subcell_xy_bounds_10",i)=bad;
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    }
    n->map_.at("collision_subcell_xy_bounds_10",i)=0xF4C0;
    for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    EXPECT_EQ(n->collisionSubcellBits(n->map_,i),(1<<10)|(1<<14));EXPECT_DOUBLE_EQ(n->footprint_margin_,.1);
  }
  static void global_future_reverse_is_never_smuggled_into_a_forward_guide()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->reverse_enable_=true;
    OdomState odom;odom.valid=true;Pose2 active;std::vector<Pose2> guide,prefix;
    ASSERT_TRUE(n->resolveTopologyGoalLocked(odom,Pose2{-2,0,0,0},active,guide));
    ASSERT_GT(guide.size(),2u);
    // A forward-only route may actually turn around and reach negative X.
    // Its INITIAL motion must still be forward; reaching that side must be
    // a real bounded-curvature turn, not the old truncated future gear plan.
    EXPECT_GT(guide[1].x,guide[0].x);
    EXPECT_TRUE(n->bodyArcPathFeasible(guide));
    EXPECT_GT(guide.back().s,std::hypot(guide.back().x,guide.back().y)+.5);
    for(size_t i=1;i<guide.size();++i) {
      EXPECT_EQ(guide[i].direction,1);
      const double dot=(guide[i].x-guide[i-1].x)*std::cos(guide[i-1].yaw)+
        (guide[i].y-guide[i-1].y)*std::sin(guide[i-1].yaw);
      EXPECT_GT(dot,0.);
    }
    EXPECT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));
  }
  static void impossible_odometry_cannot_be_clamped_into_a_valid_control_state()
  {
    for(int mode=0;mode<5;++mode) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;n->max_speed_=2.;
      auto msg=std::make_shared<nav_msgs::msg::Odometry>();
      msg->header.frame_id="map";msg->header.stamp.sec=10;msg->pose.pose.orientation.w=1;
      n->odomCallback(msg);ASSERT_TRUE(n->odom_.valid);
      msg->header.stamp.nanosec=100000000;
      msg->pose.pose.position.x=.1;msg->pose.pose.position.z=.04;n->odomCallback(msg);
      ASSERT_FALSE(n->localization_fault_);ASSERT_TRUE(n->odom_.valid);
      const auto epoch=n->input_epoch_.load();
      msg->header.stamp.nanosec=200000000;
      if(mode==0)msg->pose.pose.position.x=5.;
      if(mode==1)msg->pose.pose.position.z=NAN;
      if(mode==2)msg->pose.pose.orientation.w=0.;
      if(mode==3){msg->header.stamp.nanosec=100000000;msg->pose.pose.position.x=.2;}
      if(mode==4){msg->pose.pose.orientation.w=0.;msg->pose.pose.orientation.z=1.;}
      n->odomCallback(msg);
      EXPECT_TRUE(n->localization_fault_)<<mode;EXPECT_FALSE(n->odom_.valid);
      EXPECT_GT(n->input_epoch_.load(),epoch);EXPECT_DOUBLE_EQ(n->odom_.x,.1);
      n->have_goal_=true;n->goal_seq_=3;
      EXPECT_FALSE(n->planSnapshotCurrent(3,n->input_epoch_.load()));
      geometry_msgs::msg::Twist command;command.linear.x=.7;command.angular.z=.3;
      n->publishCommand(command);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);EXPECT_DOUBLE_EQ(n->last_cmd_steer_,0.);
      n->controlTimer();EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      msg->header.stamp.nanosec=300000000;msg->pose.pose.position.x=.2;
      msg->pose.pose.position.z=0.;msg->pose.pose.orientation.w=1.;msg->pose.pose.orientation.z=0.;
      n->odomCallback(msg);EXPECT_FALSE(n->odom_.valid);EXPECT_TRUE(n->localization_fault_);
    }
  }
  static void historical_start_relief_needs_a_current_independent_state_certificate()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    EXPECT_TRUE(n->topologyAdvisoryStartQualified(0,0,0));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->topologyAdvisoryStartQualified(0,0,0));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(.1F);
    EXPECT_FALSE(n->topologyAdvisoryStartQualified(0,0,0));
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(1e-8F);
    n->live_obstacle_enable_=true;
    EXPECT_FALSE(n->topologyAdvisoryStartQualified(0,0,0)); // missing live input
    EXPECT_DOUBLE_EQ(n->chance_state_limit_,.05);
  }
  static void supported_short_topology_prefix_is_not_discarded_by_a_long_endpoint_threshold()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;
    for(int k=0;k<n->map_.getSizeYaw();++k)
      for(int r=0;r<n->map_.getSizePos()[0];++r)for(int c=0;c<n->map_.getSizePos()[1];++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,k),p);
        if(p.x()>.45)n->map_["probability_valid"][k](r,c)=0.;
      }
    OdomState odom;Pose2 active;std::vector<Pose2> guide,prefix;
    ASSERT_TRUE(n->resolveTopologyGoalLocked(odom,Pose2{3,0,0,0},active,guide));
    EXPECT_GE(active.x,.25);EXPECT_LE(active.x,.45);
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    EXPECT_GE(prefix.back().s,.25);EXPECT_LE(prefix.back().s,.45);
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->resolveTopologyGoalLocked(odom,Pose2{3,0,0,0},active,guide));
  }
  static void routing_history_preserves_a_wall_when_current_floor_returns_do_not_see_it()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->map_stamp_=n->now();
    n->map_.add("terrain_observed",false,1.F);
    n->map_.add("historical_obstacle_advisory",false,0.F);
    n->map_.add("historical_obstacle_source_valid",false,1.F);
    n->map_.add("historical_obstacle_source_age_s",false,.2F);
    Eigen::Array3i wall;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.3,1.3,0.),wall));
    n->map_.at("historical_obstacle_advisory",wall)=1.F;
    Eigen::Vector3d centre;n->map_.index2Pos(wall,centre);
    const double wx=centre.x()-.05,wy=centre.y()-.05;
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=n->min_x_;live->min_y=n->min_y_;live->resolution=.1;
    live->width=n->width_*2;live->height=n->height_*2;live->stamp_sec=n->now().seconds();
    live->uninflated_present.assign(live->width*live->height,0);
    live->hard=live->present_hard=live->uninflated_present;
    live->routing_clear.assign(live->width*live->height,1);
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    OdomState odom;Pose2 active;std::vector<Pose2> guide;
    const auto refresh=[&]{n->resolveTopologyGoalLocked(odom,Pose2{3,0,0,0},active,guide,true);};
    const auto before=n->queryCostLocked(wx,wy,0);
    refresh();EXPECT_TRUE(n->route_memory_.occupiedAt({wx,wy}));
    const auto after=n->queryCostLocked(wx,wy,0);
    EXPECT_EQ(after.probability_upper,before.probability_upper);EXPECT_EQ(after.lethal,before.lethal);
    // Mapper not available: an empty advisory layer cannot clear old geometry.
    n->map_["historical_obstacle_advisory"][0].setZero();
    n->map_["historical_obstacle_source_valid"][0].setZero();
    refresh();EXPECT_TRUE(n->route_memory_.occupiedAt({wx,wy}));
    n->map_["historical_obstacle_source_valid"][0].setOnes();
    n->map_["historical_obstacle_source_age_s"][0].setConstant(2.);
    refresh();EXPECT_TRUE(n->route_memory_.occupiedAt({wx,wy}));
    // A fresh ray-cleared mapper snapshot AND current floor evidence permits
    // prompt routing refresh. The separate live collision guard is untouched.
    n->map_["historical_obstacle_source_age_s"][0].setConstant(.2);
    refresh();EXPECT_FALSE(n->route_memory_.occupiedAt({wx,wy}));
  }
  static void live_routing_clearance_uses_actual_distinct_scans_and_preserves_occlusion()
  {
    auto n=node(.001F);n->live_obstacle_resolution_=.1;n->live_obstacle_min_hits_=1;
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    auto geometry=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    geometry->grid.add("terrain_observed",false,1.F);
    geometry->grid.add("gp_elevation_observed",false,0.F);
    const auto scan=[&](double stamp,float elevation,bool measured=true) {
      rcl_set_ros_time_override(clock,static_cast<int64_t>(stamp*1e9));
      geometry->stamp=n->now();geometry->grid["terrain_observed"][0].setConstant(measured?1:0);
      std::atomic_store(&n->control_map_snapshot_,std::make_shared<const ControlMapSnapshot>(*geometry));
      auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
      cloud->header.frame_id="map";cloud->header.stamp=n->now();
      sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(1);
      sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
      *x=1.05F;*y=.05F;*z=elevation;n->cloudCallback(cloud);
      return std::atomic_load(&n->live_snapshot_);
    };
    auto live=scan(10.,.6F);ASSERT_TRUE(live);
    const int cell=int(std::floor((.05-live->min_y)/live->resolution))*live->width+
                   int(std::floor((1.05-live->min_x)/live->resolution));
    EXPECT_TRUE(live->uninflated_present[cell]);EXPECT_FALSE(live->routing_clear[cell]);
    EXPECT_FALSE(scan(10.1,0.F)->routing_clear[cell]);
    EXPECT_FALSE(scan(10.1,0.F)->routing_clear[cell]); // duplicate is not a second witness
    EXPECT_FALSE(scan(10.2,0.F)->routing_clear[cell]); // current protective hold still wins
    EXPECT_EQ(n->live_routing_free_scans_[cell],2);
    EXPECT_FALSE(scan(10.3,NAN)->routing_clear[cell]); // no observation is no clearance
    EXPECT_FALSE(scan(10.4,0.F,false)->routing_clear[cell]);
    EXPECT_FALSE(scan(10.5,0.F,false)->routing_clear[cell]); // GP-only support
    EXPECT_FALSE(scan(10.6,.6F)->routing_clear[cell]);
    EXPECT_FALSE(scan(12.,0.F)->routing_clear[cell]); // gap: a new pair is required
    EXPECT_TRUE(scan(12.1,0.F)->routing_clear[cell]);
  }
  static void continuous_collision_composes_the_height_bound_without_rounded_pose_veto()
  {
    for(bool rolling:{false,true}) {
      auto n=node(1e-8F);
      n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
      n->footprint_margin_=.10;
      if(rolling)n->map_.setStartIndex(Eigen::Array2i(3,7));
      n->map_.add("collision_subcell_contract",false,3.);
      n->map_.add("collision_subcell_factor",false,2.);
      n->map_.add("collision_subcell_bits",false,0.);
      const auto body=n->collisionEnvelope(n->footprint_margin_);
      for(const auto &p:std::vector<std::pair<std::string,double>>{
          {"collision_body_front",body.front},{"collision_body_back",body.back},{"collision_body_half_width",body.half_width}})
        n->map_.add(p.first,false,p.second);
      EXPECT_FALSE(n->probabilityContractValid(n->map_)); // no independent height bound
      n->map_.add("terrain_failure_probability_upper",true,1e-8);
      ASSERT_TRUE(n->probabilityContractValid(n->map_));
      // A representative grid pose may collide while the actual pose does
      // not. The display overlay must never replace the independent proof.
      for(auto &layer:n->map_["failure_probability_upper"])layer.setOnes();
      Eigen::Array3i index;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.7,.3,0.),index));
      n->map_.at("collision_risk",index)=1.;n->map_.at("collision_subcell_bits",index)=2.;
      n->rebuildCostmapLocked();
      auto clear=n->queryCostLocked(0,0,0);EXPECT_TRUE(clear.known);EXPECT_FALSE(clear.lethal);
      EXPECT_NEAR(clear.probability_upper,1e-8,1e-12);
      const double guide_cost=n->topologyTerrainCostLocked(0,0,0);
      for(auto &layer:n->map_["failure_probability_upper"])layer.setZero();
      EXPECT_DOUBLE_EQ(n->topologyTerrainCostLocked(0,0,0),guide_cost);
      for(auto &layer:n->map_["failure_probability_upper"])layer.setOnes();
      double cost=0.;EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
      auto hit=n->queryCostLocked(.05,0,0);EXPECT_TRUE(hit.known);EXPECT_TRUE(hit.lethal);
      EXPECT_DOUBLE_EQ(hit.probability_upper,1.);
      EXPECT_FALSE(n->footprintCostLocked(.05,0,0,cost));
      n->map_.at("collision_subcell_bits",index)=0.; // missing witness cannot clear the square
      EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
      n->map_.at("collision_subcell_bits",index)=2.;
      n->map_.at("collision_risk",index)=NAN;
      auto unknown=n->queryCostLocked(0,0,0);EXPECT_FALSE(unknown.known);EXPECT_TRUE(unknown.lethal);
      n->map_.at("collision_risk",index)=1.;
      for(auto &layer:n->map_["probability_valid"])layer.setZero();
      unknown=n->queryCostLocked(0,0,0);EXPECT_FALSE(unknown.known);EXPECT_TRUE(unknown.lethal);
      EXPECT_DOUBLE_EQ(n->chance_state_limit_,.05);EXPECT_DOUBLE_EQ(n->footprint_margin_,.10);
    }
  }
  static void fine_pose_contract_cannot_use_a_smaller_body_or_omit_its_geometry()
  {
    auto n=node(1e-8F);ASSERT_TRUE(n->probabilityContractValid(n->map_));
    n->map_.add("collision_subcell_contract",false,2.);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    const auto body=n->collisionEnvelope(n->footprint_margin_);
    for(const auto &p:std::vector<std::pair<std::string,double>>{
        {"collision_body_front",body.front},{"collision_body_back",body.back},{"collision_body_half_width",body.half_width}})
      n->map_.add(p.first,false,p.second);
    EXPECT_TRUE(n->probabilityContractValid(n->map_));
    n->map_["collision_body_half_width"][0].setConstant(body.half_width-.01);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
  }
  static void fine_collision_contract_refines_only_geometry_and_fails_closed_on_bad_data()
  {
    for(bool rolling:{false,true}) {
      auto n=node(1e-8F);
      n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
      n->footprint_margin_=.10;
      if(rolling)n->map_.setStartIndex(Eigen::Array2i(3,7));
      Eigen::Array3i index;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.7,.3,0.),index));
      n->map_.at("collision_risk",index)=1.;
      n->rebuildCostmapLocked();double cost=0;
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost)); // legacy full square
      n->map_.add("collision_subcell_contract",false,1.);
      n->map_.add("collision_subcell_factor",false,2.);
      n->map_.add("collision_subcell_bits",false,0.);
      n->map_.at("collision_subcell_bits",index)=2.;
      EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
      EXPECT_FALSE(n->footprintCostLocked(.05,0,0,cost)); // same measured obstacle still blocks
      EXPECT_FALSE(n->queryCostLocked(.65,.25,0.).lethal);
      EXPECT_TRUE(n->queryCostLocked(.75,.25,0.).lethal);
      for(double invalid:{0.,16.,2.5,std::numeric_limits<double>::quiet_NaN()}) {
        n->map_.at("collision_subcell_bits",index)=invalid;
        EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
      }
      n->map_.at("collision_subcell_bits",index)=2.;
      n->map_.at("collision_subcell_factor",index)=3.;
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
      n->map_.at("collision_subcell_factor",index)=2.;
      for(auto &layer:n->map_["probability_valid"])layer.setZero();
      EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost)); // geometry refinement cannot create support
      EXPECT_DOUBLE_EQ(n->footprint_margin_,.10);
      EXPECT_DOUBLE_EQ(n->chance_state_limit_,.05);
    }
  }
  static void ordinary_certified_forward_motion_can_release_an_invalidated_exit_retry_guard()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->have_goal_=true;n->goal_seq_=1;
    n->raw_goal_={3,0,0,0};n->max_odom_age_sec_=2.;n->reverse_recovery_count_=2;n->reverse_recovery_total_=7;
    auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(
      TerrainNavNode::RecoveryProgressOrigin{0,0,1,n->input_epoch_.load()});
    std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
    const auto start=n->now()-rclcpp::Duration::from_seconds(1.);
    OdomState odom;odom.valid=true;odom.v=.5;n->last_cmd_v_=.5;
    for(int k=0;k<=8;++k) {
      odom.x=.1*k;odom.stamp=start+rclcpp::Duration::from_seconds(.1*k);
      n->recordCertifiedRecoveryForwardProgress(odom,true);
      for(int repeat=0;repeat<4;++repeat)n->recordCertifiedRecoveryForwardProgress(odom,true);
    }
    const auto progress=std::atomic_load(&n->recovery_forward_progress_);
    ASSERT_TRUE(progress);EXPECT_NEAR(progress->distance,.8,1e-9);
    // The former recovery path is GONE. Its replacement is ordinary forward
    // planning, but real certified progress still releases only the retry cap.
    n->recovery_path_=false;n->recovery_exit_active_=false;n->odom_=odom;
    ASSERT_TRUE(n->planPath("ordinary replan after invalidated recovery exit"));
    EXPECT_EQ(n->reverse_recovery_count_,0);EXPECT_EQ(n->reverse_recovery_total_,7);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
  }
  static void control_pose_prediction_must_not_replace_source_recovery_progress()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->max_odom_age_sec_=2.;n->last_cmd_v_=.5;
    auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(
      TerrainNavNode::RecoveryProgressOrigin{0,0,1,n->input_epoch_.load()});
    std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
    const auto start=n->now()-rclcpp::Duration::from_seconds(1.);
    for(int k=0;k<=8;++k) {
      OdomState source;source.valid=true;source.v=.5;source.x=.1*k;
      source.stamp=start+rclcpp::Duration::from_seconds(.1*k);n->odom_=source;
      for(int repeat=0;repeat<5;++repeat) {
        auto predicted=source;predicted.x+=.02*repeat;
        n->publishChanceStatus(predicted,1e-8,true,"tracking");
      }
    }
    const auto progress=std::atomic_load(&n->recovery_forward_progress_);
    ASSERT_TRUE(progress);EXPECT_NEAR(progress->distance,.8,1e-9);
    EXPECT_NEAR(n->recovery_forward_last_.x,n->odom_.x,1e-9);
  }
  static void recovery_progress_cannot_count_commands_reverse_unknown_or_obsolete_origin()
  {
    for(int mode=0;mode<6;++mode) {
      auto n=node(1e-8F);n->topology_guide_enable_=true;n->max_odom_age_sec_=2.;
      auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(
        TerrainNavNode::RecoveryProgressOrigin{0,0,1,n->input_epoch_.load()});
      std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
      OdomState odom;odom.valid=true;odom.v=.5;n->last_cmd_v_=mode==2?-.5:.5;
      const auto start=n->now()-rclcpp::Duration::from_seconds(1.);
      if(mode==3)for(auto &layer:n->map_["probability_valid"])layer.setZero();
      if(mode==3){auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
        map->grid=n->map_;std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));}
      for(int k=0;k<=8;++k) {
        odom.x=mode==0?0:mode==1?-.1*k:mode==5?1.5*(k%2):.1*k;
        odom.stamp=start+rclcpp::Duration::from_seconds(.1*k);
        n->recordCertifiedRecoveryForwardProgress(odom,mode!=4);
      }
      const auto progress=std::atomic_load(&n->recovery_forward_progress_);
      ASSERT_TRUE(progress);EXPECT_DOUBLE_EQ(progress->distance,0.)<<mode;
    }
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->reverse_recovery_count_=2;
    auto before=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(TerrainNavNode::RecoveryProgressOrigin{0,0,1,0});
    auto newer=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(*before);
    std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(newer));
    OdomState odom;odom.x=1.;n->completeTopologyRecovery(1,0,odom,before);
    EXPECT_EQ(n->reverse_recovery_count_,2); // same mission, DIFFERENT maneuver
    n->completeTopologyRecovery(1,0,odom,newer);EXPECT_EQ(n->reverse_recovery_count_,0);
  }
  static void recovery_search_turn_is_consumed_once_and_scoped_to_goal_and_input()
  {
    auto n=node(1e-8F);n->have_goal_=true;n->goal_seq_=7;n->reverse_recovery_enable_=true;
    const auto epoch=n->input_epoch_.load();
    n->deferRecoverySearch(7,epoch);
    EXPECT_TRUE(n->consumeRecoverySearchTurn(7,epoch));
    EXPECT_FALSE(n->consumeRecoverySearchTurn(7,epoch));
    n->deferRecoverySearch(7,epoch);
    EXPECT_FALSE(n->consumeRecoverySearchTurn(8,epoch));
    EXPECT_FALSE(n->consumeRecoverySearchTurn(7,epoch));
    n->deferRecoverySearch(7,epoch);
    EXPECT_FALSE(n->consumeRecoverySearchTurn(7,epoch+1));
    n->deferRecoverySearch(6,epoch);
    EXPECT_FALSE(std::atomic_load(&n->pending_recovery_search_));
    ++n->input_epoch_;n->deferRecoverySearch(7,epoch);
    EXPECT_FALSE(std::atomic_load(&n->pending_recovery_search_));
    n->deferRecoverySearch(7,n->input_epoch_.load());
    EXPECT_TRUE(n->consumeRecoverySearchTurn(7,n->input_epoch_.load()));
    n->reverse_recovery_enable_=false;n->deferRecoverySearch(7,n->input_epoch_.load());
    EXPECT_FALSE(std::atomic_load(&n->pending_recovery_search_));
  }
  static void exhausted_shared_job_does_not_publish_path_or_start_recovery()
  {
    auto n=node(1e-8F);n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};
    n->odom_.valid=true;n->odom_.stamp=n->now();n->reverse_recovery_enable_=true;
    const auto sequence=n->path_seq_;
    PlanningBudget exhausted(0);
    EXPECT_FALSE(n->planPath("expired qualification"));
    EXPECT_FALSE(n->planReverseRecovery(1,n->input_epoch_.load()));
    EXPECT_FALSE(n->executePlannerJob("expired qualification"));
    EXPECT_EQ(n->path_seq_,sequence);EXPECT_FALSE(n->have_path_);
    EXPECT_EQ(n->reverse_recovery_total_,0);
    EXPECT_FALSE(std::atomic_load(&n->pending_recovery_search_));
  }
  static void high_probability_upper_bound_is_soft_routing_but_hard_execution()
  {
    auto n=node(.8F);n->topology_guide_enable_=true;
    EXPECT_TRUE(std::isfinite(n->topologyTerrainCostLocked(0,0,0)));
    EXPECT_GT(n->topologyTerrainCostLocked(0,0,0),3.);
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
    std::vector<Pose2> path={{0,0,0,0},{1,0,0,1}};
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_DOUBLE_EQ(n->topologyTerrainCostLocked(0,0,0),8.+std::max(0.,n->comfort_clearance_weight_));
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
  }
  static void path_retention_requires_current_entire_body_and_joint_risk_not_old_certificate()
  {
    auto n=node(1e-8F);std::vector<Pose2> path;
    for(int i=0;i<=24;++i)path.push_back(Pose2{.1*i,0,0,.1*i});
    OdomState odom;odom.x=.5;odom.y=.02;odom.yaw=.01;
    EXPECT_TRUE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
    Pose2 active;std::string reason;std::vector<Pose2> guide;
    ASSERT_TRUE(n->resolveActiveGoalLocked(odom,Pose2{-6,0,0,0},active,reason,guide,path,true));
    EXPECT_EQ(reason,"retain certified current path");EXPECT_NEAR(active.x,path.back().x,1e-9);
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,true));
    odom.x=1.5;EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
    odom.x=.5;odom.y=.4;EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
    odom.y=.02;
    auto reverse=path;for(auto &p:reverse)p.direction=-1;
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,reverse,false));
    Eigen::Array3i obstacle;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.8,0.,0.),obstacle));
    n->map_.at("collision_risk",obstacle)=1;
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
    n->resolveActiveGoalLocked(odom,Pose2{-6,0,0,0},active,reason,guide,path,true);
    EXPECT_NE(reason,"retain certified current path");
    n=node(1e-8F);for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
    n=node(.04F);EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,path,false));
  }
  static void topology_continuity_is_a_finite_preference_and_never_a_certificate()
  {
    auto n=node(1e-8F);OdomState odom;
    const std::vector<Pose2> old={{0,0,0,0},{1,0,0,1},{2,0,0,2},{3,0,0,3}};
    const auto reference=n->topologyContinuationReference(odom,old);ASSERT_GE(reference.size(),2u);
    EXPECT_DOUBLE_EQ(n->topologyContinuationPenalty(odom,reference,1.5,0),0.);
    EXPECT_GT(n->topologyContinuationPenalty(odom,reference,1.5,1.),0.);
    EXPECT_LT(n->topologyContinuationPenalty(odom,reference,1.5,100.),3.);
    EXPECT_DOUBLE_EQ(n->topologyContinuationPenalty(odom,reference,8.,1.),0.);
    EXPECT_TRUE(n->topologyContinuationReference(odom,{}).empty());
    auto reverse=old;for(auto &p:reverse)p.direction=-1;
    EXPECT_TRUE(n->topologyContinuationReference(odom,reverse).empty());
    odom.y=2.;EXPECT_TRUE(n->topologyContinuationReference(odom,old).empty());odom.y=0.;
    // Even a zero-cost reference cannot permit an obstacle, lost ground or
    // an excessive accumulated conditional upper bound.
    Eigen::Array3i hit;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.5,0,0),hit));
    n->map_.at("collision_risk",hit)=1.;EXPECT_FALSE(n->pathFootprintSafeLocked(old,.025));
    n=node(1e-8F);for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->pathFootprintSafeLocked(old,.025));
    n=node(.04F);EXPECT_FALSE(n->chancePathSafe(old,nullptr,.1));
  }
  static void early_continuation_search_cannot_discard_a_better_current_certificate()
  {
    auto n=node(1e-8F);OdomState odom;odom.x=.2;
    std::vector<Pose2> old;for(int i=0;i<=20;++i)old.push_back(Pose2{.1*i,0,0,.1*i});
    const std::vector<Pose2> short_path={{.2,0,0,0},{.7,0,0,.5}};
    const Pose2 goal{6,0,0,0};
    Eigen::Array3i obstacle;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(3,0,0),obstacle));
    n->map_.at("collision_risk",obstacle)=1.;n->rebuildCostmapLocked();
    ASSERT_TRUE(n->pathFootprintSafeLocked(old,.025));
    EXPECT_TRUE(n->canRetainCertifiedForwardPathLocked(odom,old,false));
    EXPECT_TRUE(n->continuationSearchUsefulLocked(odom,old)); // request early search, not revoke proof
    // A wall beyond the finite endpoint requests an early search but cannot
    // erase the finite preference for an actually clear slalom entrance.
    const auto continuation=n->topologyContinuationReference(odom,old);
    EXPECT_GE(continuation.size(),2u);
    EXPECT_DOUBLE_EQ(n->topologyContinuationPenalty(odom,continuation,.5,0.),0.);
    const double away=n->topologyContinuationPenalty(odom,continuation,.5,1.);
    EXPECT_GT(away,0.);EXPECT_LE(away,2.*n->guide_follow_weight_);
    EXPECT_DOUBLE_EQ(away,TerrainNavNode::continuationPenalty(odom,continuation,.5,1.,n->guide_follow_weight_));
    EXPECT_TRUE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,false));
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,old,false));
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,true));
    odom.x=1.;EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,false));odom.x=.2;
    // A real hazard ON the remaining path, unknown support or excessive
    // conditional risk revokes the old prefix just as it revokes a new one.
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.,0,0),obstacle));n->map_.at("collision_risk",obstacle)=1.;
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,false));
    const auto truncated=n->topologyContinuationReference(odom,old);
    EXPECT_TRUE(truncated.empty() || truncated.back().x<1.);
    n=node(1e-8F);for(auto &v:n->map_["probability_valid"])v.setZero();
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,false));
    n=node(.04F);EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,goal,old,short_path,false));
  }

  static void a_longer_new_guide_cannot_flip_a_still_certified_near_approach()
  {
    auto n=node(1e-8F);n->rear_axle_offset_=.215;n->live_obstacle_enable_=false;
    OdomState odom;odom.valid=true;odom.v=.25;const Pose2 mission{6,0,0,0};
    const auto curve=[&](double curvature,double length) {
      std::vector<Pose2> path;
      for(double s=0;s<length+1e-8;s+=.05)path.push_back(n->advanceArc(Pose2{},curvature,s));
      n->updatePathArcLength(path);return path;
    };
    auto old=curve(-.4,2.5),opposite=curve(.4,3.),extension=curve(-.4,3.);
    ASSERT_TRUE(n->bodyArcPathFeasible(old));ASSERT_TRUE(n->bodyArcPathFeasible(opposite));
    ASSERT_TRUE(n->pathFootprintSafeLocked(old,.025));ASSERT_TRUE(n->pathFootprintSafeLocked(opposite,.025));
    ASSERT_TRUE(n->canRetainCertifiedForwardPathLocked(odom,old,false));
    ASSERT_GT(opposite.back().s,old.back().s); // old shorter-prefix heuristic did not protect this
    EXPECT_TRUE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,false));
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,extension,false));
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,true));
    // Once the useful old horizon is consumed, no sticky branch commitment.
    const auto near_end=n->samplePathAt(old,1.5);odom.x=near_end.x;odom.y=near_end.y;odom.yaw=near_end.yaw;
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,false));
    odom=OdomState{};odom.valid=true;
    Eigen::Array3i obstacle;const auto hit=n->samplePathAt(old,1.5);
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(hit.x,hit.y,hit.yaw),obstacle));
    n->map_.at("collision_risk",obstacle)=1.;
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,false));
    n->map_.at("collision_risk",obstacle)=0.;
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,false));
    for(auto &layer:n->map_["probability_valid"])layer.setOnes();
    for(auto &layer:n->map_["failure_probability_upper"])layer.setConstant(.04F);
    EXPECT_FALSE(n->preferCertifiedRemainingPathLocked(odom,mission,old,opposite,false));
  }

  static void tangent_obstacle_is_a_search_hint_not_a_finite_curve_collision()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    n->rear_axle_offset_=.2078;
    std::vector<Pose2> curve;
    for(int i=0;i<=40;++i)curve.push_back(n->advanceArc(Pose2{},.35,i*.05));
    n->updatePathArcLength(curve);
    const auto endpoint=curve.back();Eigen::Array3i obstacle;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(endpoint.x+.95*std::cos(endpoint.yaw),
      endpoint.y+.95*std::sin(endpoint.yaw),0.),obstacle));
    n->map_.at("collision_risk",obstacle)=1.;n->rebuildCostmapLocked();
    OdomState odom;odom.valid=true;odom.stamp=n->now();
    ASSERT_TRUE(n->bodyArcPathFeasible(curve));
    ASSERT_TRUE(n->pathFootprintSafeLocked(curve,.025));
    ASSERT_TRUE(n->chancePathSafe(curve,nullptr,n->chance_path_budget_));
    ASSERT_TRUE(n->continuationSearchUsefulLocked(odom,curve));
    EXPECT_TRUE(n->canRetainCertifiedForwardPathLocked(odom,curve,false));
    // With no complete replacement, an unrelated side frontier is not a
    // reason to erase a still certified turning trajectory while its whole
    // route job is pending. Without such a job, directed bypass search is
    // still required (the unknown-frontier regression checks that case).
    std::promise<TerrainNavNode::BackgroundGuideResult> pending;
    n->background_guide_=pending.get_future();
    Pose2 active;std::string reason;std::vector<Pose2> guide;
    ASSERT_TRUE(n->resolveActiveGoalLocked(odom,{6.,0.,0.,0.},active,reason,guide,curve,true));
    EXPECT_EQ(reason,"retain certified current path");
    EXPECT_NEAR(active.x,endpoint.x,1e-10);EXPECT_NEAR(active.y,endpoint.y,1e-10);
    ASSERT_EQ(guide.size(),curve.size());
    for(size_t i=0;i<curve.size();++i) {
      EXPECT_DOUBLE_EQ(guide[i].x,curve[i].x);EXPECT_DOUBLE_EQ(guide[i].yaw,curve[i].yaw);
    }
    pending.set_value(TerrainNavNode::BackgroundGuideResult{});
    // An actual obstacle ON the finite trajectory still revokes it.
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(curve[20].x,curve[20].y,0.),obstacle));
    n->map_.at("collision_risk",obstacle)=1.;
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,curve,false));
    n=node(1e-8F);for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,curve,false));
    n=node(.04F);EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(odom,curve,false));
  }

  static void search_failure_near_side_obstacle_cannot_reverse_a_certified_forward_path()
  {
    auto n=node(1e-8F);n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->live_obstacle_enable_=true;n->hard_clearance_floor_=0.;
    n->have_goal_=true;n->raw_goal_={6.,0.,0.,0.};n->odom_.valid=true;n->odom_.stamp=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=-4.;live->min_y=-4.;live->resolution=.2;live->width=live->height=40;
    live->stamp_sec=n->now().seconds();live->inflation_radius=0.;live->hard.assign(1600,0);
    const int side=int((1.1+4.)/.2)*40+int((1.+4.)/.2);live->hard[side]=1;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->path_.clear();
    for(int i=0;i<=20;++i)n->path_.push_back({i*.1,0.,0.,i*.1,1});
    n->have_path_=true;const auto before=n->path_;const auto sequence=n->path_seq_;
    ASSERT_TRUE(n->canRetainCertifiedForwardPathLocked(n->odom_,before,false));
    EXPECT_FALSE(n->planReverseRecovery());
    EXPECT_FALSE(n->recovery_path_);EXPECT_EQ(n->path_seq_,sequence);
    EXPECT_EQ(n->reverse_recovery_count_.load(),0);
    ASSERT_EQ(n->path_.size(),before.size());EXPECT_DOUBLE_EQ(n->path_.back().x,2.);
    // This is not a blanket reverse disable: invalid support revokes the
    // certificate, and the existing dead-end/blocked-front tests still need
    // to find a freshly certified retreat where one exists.
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->canRetainCertifiedForwardPathLocked(n->odom_,before,false));
  }

  static void far_path_collision_keeps_only_the_current_certified_prefix()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->hard_clearance_floor_=0.;
    std::vector<Pose2> path;for(int i=0;i<=32;++i)path.push_back({i*.1,0,0,i*.1,1});
    auto map=*std::atomic_load(&n->control_map_snapshot_);
    Eigen::Array3i hit;ASSERT_TRUE(map.grid.pos2Index(Eigen::Vector3d(2.8,0.,0.),hit));
    map.grid.at("collision_risk",hit)=1.;
    OdomState odom;std::vector<Pose2> prefix;
    ASSERT_TRUE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));
    EXPECT_GT(prefix.back().s,1.5);EXPECT_LT(prefix.back().s,2.8);
    for(const auto &p:prefix) {
      double cost=0;EXPECT_TRUE(n->footprintCostLocked(p.x,p.y,p.yaw,cost,nullptr,nullptr,false,&map));
      EXPECT_DOUBLE_EQ(p.y,0.);EXPECT_DOUBLE_EQ(p.yaw,0.);
    }
    EXPECT_TRUE(n->chancePathSafe(prefix,&map,n->chance_path_budget_));
    // Raw geometry cannot be erased, or changed by asking for a prefix.
    EXPECT_DOUBLE_EQ(map.grid.at("collision_risk",hit),1.);
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=-4.;live->min_y=-4.;live->resolution=.2;live->width=live->height=40;
    live->inflation_radius=0.;live->hard.assign(1600,0);
    live->hard[20*40+34]=1;map.grid.at("collision_risk",hit)=0.;
    EXPECT_TRUE(n->certifyUnchangedPathPrefix(odom,path,map,live.get(),prefix));
    EXPECT_GT(prefix.back().s,1.5);EXPECT_LT(prefix.back().s,2.8);
  }

  static void prefix_repair_never_bridges_unknown_or_a_current_body_collision()
  {
    auto n=node(1e-8F);n->hard_clearance_floor_=0.;n->live_obstacle_enable_=false;
    std::vector<Pose2> path;for(int i=0;i<=30;++i)path.push_back({i*.1,0,0,i*.1,1});
    auto map=*std::atomic_load(&n->control_map_snapshot_);
    for(auto &slice:map.grid["probability_valid"])
      for(int r=0;r<40;++r)for(int c=0;c<40;++c) {
        Eigen::Vector3d p;map.grid.index2Pos(Eigen::Array3i(r,c,0),p);
        if(p.x()>1.2&&p.x()<1.8)slice(r,c)=0.;
      }
    OdomState odom;std::vector<Pose2> prefix;
    ASSERT_TRUE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));
    EXPECT_LE(prefix.back().x,1.2);EXPECT_TRUE(n->chancePathSafe(prefix,&map,.1));
    Eigen::Array3i hit;ASSERT_TRUE(map.grid.pos2Index(Eigen::Vector3d(0.,0.,0.),hit));
    map.grid.at("collision_risk",hit)=1.;
    EXPECT_FALSE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));EXPECT_TRUE(prefix.empty());
    map=*std::atomic_load(&n->control_map_snapshot_);
    // No failure: do not manufacture truncation of an entirely valid curve.
    EXPECT_FALSE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));
  }

  static void repaired_prefix_requires_stop_distance_and_cannot_overwrite_a_new_path()
  {
    auto n=node(1e-8F);n->hard_clearance_floor_=0.;n->live_obstacle_enable_=false;
    n->have_path_=true;n->have_goal_=true;n->odom_.valid=true;n->odom_.stamp=n->now();
    for(int i=0;i<=30;++i)n->path_.push_back({i*.1,0.,0.,i*.1,1});
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    Eigen::Array3i hit;ASSERT_TRUE(map->grid.pos2Index(Eigen::Vector3d(2.8,0.,0.),hit));map->grid.at("collision_risk",hit)=1.;
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto old=n->path_;const auto sequence=n->path_seq_;
    EXPECT_FALSE(n->repairInvalidPathSuffix(n->odom_,old,sequence+1));
    EXPECT_EQ(n->path_seq_,sequence);EXPECT_DOUBLE_EQ(n->path_.back().x,3.);
    n->last_cmd_v_=2.; // momentum reserve exceeds this otherwise clear prefix
    EXPECT_FALSE(n->repairInvalidPathSuffix(n->odom_,old,sequence));
    EXPECT_EQ(n->path_seq_,sequence);n->last_cmd_v_=0.;
    ASSERT_TRUE(n->repairInvalidPathSuffix(n->odom_,old,sequence));
    EXPECT_EQ(n->safety_prefix_seq_,n->path_seq_);EXPECT_LT(n->path_.back().s,3.);
    EXPECT_TRUE(n->plan_requested_);EXPECT_TRUE(n->have_path_);EXPECT_FALSE(n->recovery_path_);
    const double distance=n->path_.back().s,latency=.7,deceleration=1.2;
    const double speed=terrain_nav::clearanceSpeedLimit(distance,.15,latency,deceleration);
    EXPECT_LE(speed*latency+speed*speed/(2*deceleration),distance-.15+1e-9);
  }
  static void finite_prefix_certificate_handles_each_gear_but_never_a_hidden_cusp()
  {
    for(int gear:{-1,1}) {
      auto n=node(1e-8F);n->hard_clearance_floor_=0.;n->live_obstacle_enable_=false;
      std::vector<Pose2> path;for(int i=0;i<=30;++i)path.push_back({gear*i*.1,0.,0.,i*.1,gear});
      auto map=*std::atomic_load(&n->control_map_snapshot_);
      Eigen::Array3i hit;ASSERT_TRUE(map.grid.pos2Index(Eigen::Vector3d(gear*2.8,0.,0.),hit));
      map.grid.at("collision_risk",hit)=1.;OdomState odom;std::vector<Pose2> prefix;
      ASSERT_TRUE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));
      EXPECT_GT(prefix.back().s,1.5);EXPECT_LT(prefix.back().s,2.8);
      EXPECT_EQ(prefix.front().direction,gear);EXPECT_TRUE(n->chancePathSafe(prefix,&map,.1));
      path.back().direction=-gear;
      EXPECT_FALSE(n->certifyUnchangedPathPrefix(odom,path,map,nullptr,prefix));
      EXPECT_TRUE(prefix.empty());
    }
  }
  static void obsolete_guide_wait_cannot_erase_a_newly_certified_path()
  {
    auto n=node(1e-8F);n->hard_clearance_floor_=0.;n->live_obstacle_enable_=false;
    n->have_goal_=true;n->have_path_=true;n->goal_seq_=1;n->odom_.valid=true;n->odom_.stamp=n->now();
    for(int i=0;i<=20;++i)n->path_.push_back({i*.1,0.,0.,i*.1,1});
    const auto old=n->path_seq_;++n->path_seq_;
    ASSERT_TRUE(n->retainFinitePathAtGuideHandoffLocked(n->odom_,1,n->input_epoch_.load(),old));
    EXPECT_TRUE(n->have_path_);EXPECT_EQ(n->path_seq_,old+1);
    // Obsolete failure is not permission to move through new unknown terrain,
    // but it still cannot erase a newer identity. Control owns that stop.
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->retainFinitePathAtGuideHandoffLocked(n->odom_,1,n->input_epoch_.load(),old));
    EXPECT_TRUE(n->have_path_);EXPECT_EQ(n->path_seq_,old+1);
    EXPECT_FALSE(n->retainFinitePathAtGuideHandoffLocked(n->odom_,1,n->input_epoch_.load(),old+1));
    EXPECT_FALSE(n->have_path_);EXPECT_EQ(n->path_seq_,old+2);
  }
  static void repaired_recovery_preserves_same_gear_tail_and_world_fixed_next_leg()
  {
    auto n=staged_fixture();n->hard_clearance_floor_=0.;n->recovery_staged_gears_=false;
    n->path_.clear();n->recovery_legs_.clear();n->recovery_exit_.clear();
    for(int i=0;i<=30;++i)n->path_.push_back({-i*.1,0.,0.,i*.1,-1});
    for(int i=0;i<=30;++i)n->recovery_exit_.push_back({-3.+i*.1,0.,0.,i*.1,1});
    const auto original_exit=n->recovery_exit_;n->reverse_recovery_count_=1;n->reverse_recovery_total_=3;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    Eigen::Array3i hit;ASSERT_TRUE(map->grid.pos2Index(Eigen::Vector3d(-2.8,0.,0.),hit));
    map->grid.at("collision_risk",hit)=1.;n->map_.at("collision_risk",hit)=1.;
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto path=n->path_;std::vector<Pose2> certificate;
    ASSERT_TRUE(n->certifyUnchangedPathPrefix(n->odom_,path,*map,nullptr,certificate));
    ASSERT_TRUE(n->repairInvalidPathSuffix(n->odom_,path,n->path_seq_));
    EXPECT_NEAR(certificate.back().s-n->path_.back().s,.15,1e-8);
    ASSERT_EQ(n->recovery_legs_.size(),2u);EXPECT_TRUE(n->recovery_staged_gears_);
    EXPECT_EQ(n->recovery_legs_[0].front().direction,-1);
    EXPECT_DOUBLE_EQ(n->recovery_legs_[0].front().x,n->path_.back().x);
    EXPECT_DOUBLE_EQ(n->recovery_legs_[0].back().x,-3.);
    ASSERT_EQ(n->recovery_legs_[1].size(),original_exit.size());
    for(size_t i=0;i<original_exit.size();++i) {
      EXPECT_DOUBLE_EQ(n->recovery_legs_[1][i].x,original_exit[i].x);
      EXPECT_DOUBLE_EQ(n->recovery_legs_[1][i].yaw,original_exit[i].yaw);
    }
    EXPECT_EQ(n->reverse_recovery_count_,1);EXPECT_EQ(n->reverse_recovery_total_,3);
    // At the stopline, the obstructed same-gear tail cannot be skipped in
    // favour of the later forward leg, even though that leg starts behind it.
    n->odom_.x=n->path_.back().x;n->odom_.v=0.;n->odom_.stamp=n->now();
    EXPECT_FALSE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    ASSERT_EQ(n->recovery_legs_.size(),2u);EXPECT_EQ(n->path_.front().direction,-1);
    // Explicit new fixture evidence clears the obstruction; now the old
    // world-fixed reverse endpoint, then its forward exit, must be used.
    n->map_.at("collision_risk",hit)=0.;
    ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,-1);EXPECT_NEAR(n->path_.back().x,-3.,1e-8);
    n->odom_.x=-3.;n->odom_.stamp=n->now();
    ASSERT_TRUE(n->activateNextStagedGearLocked(n->odom_,n->goal_seq_,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,1);EXPECT_NEAR(n->path_.back().x,0.,1e-8);
    EXPECT_TRUE(n->recovery_legs_.empty());
    EXPECT_EQ(n->reverse_recovery_count_,1);EXPECT_EQ(n->reverse_recovery_total_,3);
  }
  static void topology_recovery_certifies_both_legs_and_unknown_never_passes()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;n->reverse_enable_=true;
    std::vector<Pose2> reverse,exit;
    ASSERT_TRUE(n->buildTopologyRecoveryLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},reverse,exit));
    EXPECT_GE(reverse.back().s,.4);EXPECT_GE(exit.back().s,1.2);
    EXPECT_TRUE(n->bodyArcPathFeasible(reverse));EXPECT_TRUE(n->bodyArcPathFeasible(exit));
    auto both=reverse;both.insert(both.end(),exit.begin(),exit.end());
    EXPECT_TRUE(n->chancePathSafe(both,nullptr,.1));
    EXPECT_TRUE(n->pathFootprintSafeLocked(reverse,.025));EXPECT_TRUE(n->pathFootprintSafeLocked(exit,.025));
    // The alternative complete-gear-route search must obtain the SAME full
    // certificate; it cannot consume a retry forever or authorize a gap.
    n->topology_recovery_search_round_=1;
    ASSERT_TRUE(n->buildTopologyRecoveryLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},reverse,exit));
    both=reverse;both.insert(both.end(),exit.begin(),exit.end());
    EXPECT_TRUE(n->chancePathSafe(both,nullptr,.1));
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_FALSE(n->buildTopologyRecoveryLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},reverse,exit));
  }
  static void global_routing_reuses_soft_clearance_without_changing_probability_or_collision()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;
    const auto raw=n->map_["collision_risk"][0];const auto probability=n->queryCostLocked(0,0,0).probability_upper;
    std::fill(n->comfort_clearance_.begin(),n->comfort_clearance_.end(),2.f);
    const double far=n->topologyTerrainCostLocked(0,0,0);
    std::fill(n->comfort_clearance_.begin(),n->comfort_clearance_.end(),.6f);
    const double close=n->topologyTerrainCostLocked(0,0,0);
    EXPECT_GT(close,far);EXPECT_TRUE(std::isfinite(close));
    EXPECT_EQ(n->queryCostLocked(0,0,0).probability_upper,probability);
    EXPECT_FALSE(n->queryCostLocked(0,0,0).lethal);
    EXPECT_TRUE(n->map_["collision_risk"][0]==raw);
  }
  static void only_completed_displaced_forward_exit_resets_consecutive_recovery_limit()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;
    const auto set=[&]{auto o=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(
      TerrainNavNode::RecoveryProgressOrigin{0,0,4,2});
      std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(o));};
    set();n->reverse_recovery_count_=2;n->reverse_recovery_total_=7;
    OdomState odom;odom.x=.3;n->completeTopologyRecovery(4,2,odom);
    EXPECT_EQ(n->reverse_recovery_count_,2);
    odom.x=1.;n->completeTopologyRecovery(5,2,odom);EXPECT_EQ(n->reverse_recovery_count_,2);
    n->completeTopologyRecovery(4,3,odom);EXPECT_EQ(n->reverse_recovery_count_,2);
    n->completeTopologyRecovery(4,2,odom);EXPECT_EQ(n->reverse_recovery_count_,0);
    EXPECT_EQ(n->reverse_recovery_total_,7);EXPECT_FALSE(std::atomic_load(&n->reverse_progress_origin_));
  }
  static void topology_prefix_stops_at_unknown_and_checks_full_body()
  {
    auto n=node(1e-8F);
    std::vector<Pose2> guide,prefix;
    for(int i=0;i<=30;++i)guide.push_back(Pose2{.1*i,0,0,.1*i});
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    EXPECT_GT(prefix.back().s,2.);
    // Missing support in the middle must not be skipped even though the
    // remaining route and terminal point have very low modeled risk.
    Eigen::Array3i gap;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.5,0,0),gap));
    for(int k=0;k<n->map_.getSizeYaw();++k)n->map_.at("probability_valid",Eigen::Array3i(gap[0],gap[1],k))=0;
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    EXPECT_LT(prefix.back().x,1.5);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    n=node(1e-8F);
    Eigen::Array3i obstacle;ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(2.,.3,0),obstacle));
    n->map_.at("collision_risk",obstacle)=1.;n->rebuildCostmapLocked();
    ASSERT_TRUE(n->buildCertifiedGuidePrefixLocked(guide,prefix));
    EXPECT_LT(prefix.back().x,2.);
    EXPECT_TRUE(n->pathFootprintSafeLocked(prefix,.025));
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));
  }
  static void optimistic_topology_guide_cannot_authorize_unknown_execution()
  {
    auto n=node(1e-8F);n->topology_guide_enable_=true;
    OdomState odom;odom.valid=true;odom.stamp=n->now();
    Pose2 active;std::vector<Pose2> guide;const Pose2 goal{3.,0.,0.,0.};
    ASSERT_TRUE(n->resolveTopologyGoalLocked(odom,goal,active,guide));
    std::vector<Pose2> path;
    ASSERT_TRUE(n->buildContinuousCurvePathLocked(Pose2{0,0,0,0},active,path));
    EXPECT_TRUE(n->pathFootprintSafeLocked(path,.025));EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->resolveTopologyGoalLocked(odom,goal,active,guide));
    EXPECT_FALSE(n->pathFootprintSafeLocked(path,.025));EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    // No observed-ground layer: predictions cannot seed observed free space.
    EXPECT_EQ(n->route_memory_.size(),0u);EXPECT_EQ(n->chance_state_limit_,.05);
    EXPECT_EQ(n->chance_path_budget_,.1);EXPECT_EQ(n->chance_braking_budget_,.05);
  }
  static void final_goal_marker_is_not_replaced_by_rolling_subgoal()
  {
    auto n=node(1e-6F);
    auto witness=std::make_shared<rclcpp::Node>("separate_goal_marker_witness");
    std::vector<visualization_msgs::msg::MarkerArray> mission,local;
    auto qos=rclcpp::QoS(1).reliable().transient_local();
    auto a=witness->create_subscription<visualization_msgs::msg::MarkerArray>("/terrain_nav/active_goal",qos,
      [&](const visualization_msgs::msg::MarkerArray &m){mission.push_back(m);});
    auto b=witness->create_subscription<visualization_msgs::msg::MarkerArray>("/terrain_nav/local_goal",qos,
      [&](const visualization_msgs::msg::MarkerArray &m){local.push_back(m);});
    rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(n->get_node_base_interface());executor.add_node(witness);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();goal->header.frame_id="map";
    goal->header.stamp=n->now();goal->pose.position.x=27.;goal->pose.position.y=3.5;goal->pose.orientation.w=1.;
    n->goalCallback(goal);n->publishActiveGoal(Pose2{4.,.1,0.,0.});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while((mission.empty() || local.empty()) && std::chrono::steady_clock::now()<deadline) {
      executor.spin_some();std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_EQ(mission.size(),1u);ASSERT_EQ(local.size(),1u);
    EXPECT_DOUBLE_EQ(mission[0].markers[1].pose.position.x,27.);
    EXPECT_DOUBLE_EQ(local[0].markers[1].pose.position.x,4.);
    EXPECT_DOUBLE_EQ(n->raw_goal_.x,27.);EXPECT_DOUBLE_EQ(n->raw_goal_.y,3.5);
    EXPECT_EQ(n->goal_seq_,1u);
  }
  static void stuck_stop_keeps_status_current_and_requires_a_new_goal()
  {
    auto n=node(1e-6F);n->have_goal_=true;n->goal_seq_=7;n->stuck_goal_seq_=7;
    OdomState odom;odom.valid=true;n->last_cmd_v_=.5;
    n->stuck_start_x_=n->stuck_start_y_=0.;n->stuck_since_sec_=n->now().seconds()-5.;
    EXPECT_TRUE(n->stopIfStuck(odom));EXPECT_TRUE(n->stuck_latched_);
    EXPECT_EQ(n->last_chance_reason_,"no progress: motion inhibited");EXPECT_FALSE(n->last_chance_accepted_);
    EXPECT_FLOAT_EQ(n->last_cmd_v_,0.);
    n->last_chance_status_sec_=n->now().seconds()-1.;
    const auto previous=n->last_chance_status_sec_;
    EXPECT_TRUE(n->stopIfStuck(odom));EXPECT_GT(n->last_chance_status_sec_,previous);
    n->goal_seq_=8;
    EXPECT_FALSE(n->stopIfStuck(odom));EXPECT_FALSE(n->stuck_latched_);
  }
  static void traction_replan_revokes_stalled_and_prestop_worker_paths()
  {
    auto n=node(1e-6F);n->stuck_replan_attempts_=2;n->live_obstacle_enable_=false;
    n->set_parameter(rclcpp::Parameter("use_sim_time",true));
    auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    OdomState odom;odom.valid=true;odom.stamp=n->now();
    n->have_goal_=true;n->goal_seq_=n->stuck_goal_seq_=7;n->raw_goal_={3,0,0,0};
    n->last_cmd_v_=-.55;n->stuck_since_sec_=95.;n->have_path_=true;n->recovery_path_=true;
    n->path_={{0,0,0,0,-1},{-.5,0,0,.5,-1}};
    n->recovery_legs_.push_back(n->path_);n->recovery_resume_pending_=true;
    const auto epoch=n->input_epoch_.load();
    EXPECT_TRUE(n->stopIfStuck(odom));EXPECT_FALSE(n->stuck_latched_);
    EXPECT_TRUE(n->mobility_recovery_.active());EXPECT_EQ(n->mobility_recovery_.attempts(),1);
    EXPECT_GT(n->input_epoch_.load(),epoch);EXPECT_FALSE(n->have_path_);
    EXPECT_TRUE(n->path_.empty());EXPECT_TRUE(n->recovery_legs_.empty());EXPECT_FALSE(n->recovery_resume_pending_);
    const auto fresh=[&](double t) {
      EXPECT_EQ(rcl_set_ros_time_override(clock,int64_t(t*1e9)),RCL_RET_OK);odom.stamp=n->now();
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      map->stamp=n->now();std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    };
    fresh(100.1);odom.v=-.1;EXPECT_TRUE(n->mobilityRecoveryControl(odom));EXPECT_FALSE(n->plan_requested_);
    fresh(100.3);odom.v=0;EXPECT_TRUE(n->mobilityRecoveryControl(odom));EXPECT_FALSE(n->plan_requested_);
    // A worker may have finished a path while braking. It must not become a
    // post-stop certificate just because the physical stop has completed.
    n->have_path_=true;n->path_={{0,0,0,0},{.5,0,0,.5}};
    const auto before=n->input_epoch_.load();fresh(100.9);
    EXPECT_TRUE(n->mobilityRecoveryControl(odom));EXPECT_TRUE(n->plan_requested_);
    EXPECT_GT(n->input_epoch_.load(),before);EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
    EXPECT_FLOAT_EQ(n->last_cmd_v_,0);EXPECT_EQ(n->raw_goal_.x,3);EXPECT_EQ(n->goal_seq_,7u);
  }
  static void first_goal_ack_survives_zero_clock_and_late_discovery()
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("use_sim_time",true)});
    auto n=std::make_shared<TerrainNavNode>(options,false);
    EXPECT_EQ(n->now().nanoseconds(),0);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();
    goal->header.frame_id="map";goal->header.stamp.sec=46;goal->header.stamp.nanosec=900000000;
    goal->pose.position.x=3.;goal->pose.orientation.w=1.;n->goalCallback(goal);
    auto witness=std::make_shared<rclcpp::Node>("first_goal_ack_witness");
    std::vector<geometry_msgs::msg::PoseStamped> received;
    auto sub=witness->create_subscription<geometry_msgs::msg::PoseStamped>("/terrain_nav/received_goal",
      rclcpp::QoS(1).reliable().transient_local(),[&](const geometry_msgs::msg::PoseStamped &m){received.push_back(m);});
    rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(n);executor.add_node(witness);
    const auto wait=[&](size_t count) {
      const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
      while(received.size()<count && std::chrono::steady_clock::now()<deadline) {
        executor.spin_some();std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    };
    wait(1);ASSERT_EQ(received.size(),1u);
    EXPECT_EQ(received.back().header.stamp,goal->header.stamp);
    EXPECT_EQ(received.back().pose.position.x,3.);
    n->odom_.stamp=rclcpp::Time(47,123,RCL_ROS_TIME);
    goal->header.stamp.sec=0;goal->header.stamp.nanosec=0;goal->pose.position.x=4.;n->goalCallback(goal);
    wait(2);ASSERT_EQ(received.size(),2u);
    EXPECT_EQ(received.back().header.stamp.sec,47);EXPECT_EQ(received.back().header.stamp.nanosec,123u);
  }

  static void reliable_registered_cloud_keeps_latest_without_changing_generic_sensor_policy()
  {
    for(bool reliable:{false,true}) {
      rclcpp::NodeOptions options;
      options.parameter_overrides({rclcpp::Parameter("live_obstacle_enable",true),
        rclcpp::Parameter("live_cloud_reliable",reliable)});
      auto n=std::make_shared<TerrainNavNode>(options,false);
      ASSERT_TRUE(n->cloud_sub_);
      const auto qos=n->cloud_sub_->get_actual_qos().get_rmw_qos_profile();
      EXPECT_EQ(qos.reliability,reliable?RMW_QOS_POLICY_RELIABILITY_RELIABLE:RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
      EXPECT_EQ(qos.depth,reliable?1u:5u);
    }
  }
  static void probability_live_columns_reject_low_step_above_physical_limit()
  {
    auto n=node(.001F);n->live_obstacle_min_height_=.32;n->chance_max_step_m_=.18;
    n->live_obstacle_min_hits_=1;n->live_obstacle_inflation_radius_=0.;n->live_motion_guard_enable_=false;
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    snapshot->grid.add("terrain_observed",false,1.F);snapshot->grid.add("gp_elevation_observed",false,0.F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();cloud->header.frame_id="map";cloud->header.stamp=n->now();
    sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(1);
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    *x=-1.15F;*y=-1.35F;*z=.20F;n->cloudCallback(cloud);
    ASSERT_TRUE(std::atomic_load(&n->live_snapshot_));
    EXPECT_TRUE(std::atomic_load(&n->live_snapshot_)->occupied(-1.15,-1.35));
    EXPECT_EQ(n->chance_state_limit_,.05);EXPECT_EQ(n->chance_path_budget_,.10);
  }
  static void motion_grid_aggregation_preserves_collision_resolution_and_extent()
  {
    auto n=node(.001F);n->live_motion_min_resolution_=.20;
    EXPECT_EQ(n->motionAggregationFactor(208,208,.10),2);
    EXPECT_EQ(n->motionAggregationFactor(104,104,.20),1);
    EXPECT_EQ(n->motionAggregationFactor(209,209,.10),1);
    n->live_motion_min_resolution_=0.;EXPECT_EQ(n->motionAggregationFactor(208,208,.10),1);
    n->live_motion_min_resolution_=.20;n->live_obstacle_resolution_=.10;
    n->live_obstacle_min_hits_=1;n->live_obstacle_inflation_radius_=0.;
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    snapshot->grid.add("terrain_observed",false,1.F);snapshot->grid.add("gp_elevation_observed",false,0.F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();cloud->header.frame_id="map";cloud->header.stamp=n->now();
    sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(1);
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    *x=-1.15F;*y=-1.35F;*z=.55F;n->cloudCallback(cloud);
    auto live=std::atomic_load(&n->live_snapshot_);ASSERT_TRUE(live);
    EXPECT_NEAR(live->resolution,.10,1e-6);EXPECT_TRUE(live->occupied(-1.15,-1.35));
    EXPECT_EQ(std::count(live->uninflated_present.begin(),live->uninflated_present.end(),1),1);
    EXPECT_EQ(live->width%2,0);EXPECT_EQ(live->height%2,0);
  }
  static void early_partial_requires_useful_existing_frontier_and_explicit_budget()
  {
    auto n=node(.001F);n->partial_search_time_ms_=20.;
    const Pose2 start{0,0,0,0},goal{5,0,0,0};
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{1,0,0,1},19.99));
    EXPECT_TRUE(n->partialFrontierReady(start,goal,Pose2{1,0,0,1},20.));
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{0,0,0,1},20.));
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{-.9,0,0,1},20.));
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{.5,0,0,.5},20.));
    EXPECT_FALSE(n->partialFrontierReady(start,Pose2{1.5,0,0,0},Pose2{1,0,0,1},20.));
    n->accept_partial_progress_=false;
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{1,0,0,1},20.));
    n->accept_partial_progress_=true;n->partial_search_time_ms_=0;
    EXPECT_FALSE(n->partialFrontierReady(start,goal,Pose2{1,0,0,1},20.));
  }
  static void fine_live_grid_preserves_extent_body_checks_and_terrain_generation()
  {
    auto n=node(.001F);n->live_obstacle_resolution_=.1;n->live_obstacle_min_hits_=1;
    auto geometry=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    geometry->grid.add("terrain_observed",false,1.F);
    geometry->grid.add("gp_elevation_observed",false,0.F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(geometry));
    const auto count=geometry->grid.getSizePos();
    auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id="map";cloud->header.stamp=n->now();
    sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(1);
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    *x=1.05F;*y=.05F;*z=.6F;n->cloudCallback(cloud);
    const auto fine=std::atomic_load(&n->live_snapshot_);ASSERT_TRUE(fine);
    EXPECT_EQ(fine->width,2*geometry->width);EXPECT_EQ(fine->height,2*geometry->height);
    EXPECT_DOUBLE_EQ(fine->resolution,.5*geometry->resolution);
    EXPECT_DOUBLE_EQ(fine->width*fine->resolution,geometry->width*geometry->resolution);
    EXPECT_TRUE(n->liveBodyUnsafe(1.05,.05,0,*fine));
    EXPECT_EQ(std::atomic_load(&n->control_map_snapshot_).get(),geometry.get());
    EXPECT_TRUE((geometry->grid.getSizePos()==count).all());
    n->live_obstacle_resolution_=0;n->cloudCallback(cloud);
    EXPECT_EQ(std::atomic_load(&n->live_snapshot_)->width,geometry->width);
  }
  static void indexed_queries_match_nearest_with_rolling_buffers()
  {
    auto n=node(.001F);
    const auto shape=n->map_.getSizePos();
    for(int k=0;k<n->map_.getSizeYaw();++k)for(int r=0;r<shape[0];++r)for(int c=0;c<shape[1];++c) {
      n->map_["failure_probability_upper"][k](r,c)=.001F*((13*r+7*c+k)%90);
      n->map_["probability_valid"][k](r,c)=(r+c+k)%11 ? 1.F : 0.F;
    }
    n->map_["collision_risk"][0](3,5)=.5F;
    n->map_["collision_risk"][0](4,5)=std::numeric_limits<float>::quiet_NaN();
    for(const auto shift:{Eigen::Array2i(0,0),Eigen::Array2i(7,11),Eigen::Array2i(39,38)}) {
      n->map_.setStartIndex(shift);n->rebuildCostmapLocked();
      for(double x=-3.7;x<3.8;x+=.43)for(double y=-3.7;y<3.8;y+=.67)
        for(double yaw:{-6.3,-3.14,-.1,0.,.11,1.7,3.14,6.3}) {
          const Eigen::Vector3d position(x,y,yaw);
          const auto sample=[&](const char *name){return n->map_.getValue(name,position,se2_grid::InterpolationMethods::INTER_NEAREST);};
          const double collision=sample("collision_risk");
          const auto q=n->queryCostLocked(x,y,yaw);
          if(!std::isfinite(collision) || collision>=.5) {
            EXPECT_TRUE(q.lethal);EXPECT_EQ(q.known,std::isfinite(collision));continue;
          }
          const double valid=sample("probability_valid"),p=sample("failure_probability_upper");
          EXPECT_EQ(q.known,std::isfinite(valid)&&valid>=.5&&std::isfinite(p)&&p>=0&&p<=1);
          EXPECT_EQ(q.lethal,!q.known||p>n->chance_state_limit_);
          if(q.known)EXPECT_DOUBLE_EQ(q.probability_upper,p);
        }
    }
    for(const auto pose:{Eigen::Vector3d(0,0,std::numeric_limits<double>::quiet_NaN()),
                         Eigen::Vector3d(10,0,0),Eigen::Vector3d(0,10,0)})
      EXPECT_TRUE(n->queryCostLocked(pose.x(),pose.y(),pose.z()).lethal);
  }
  static void map_ingress_cannot_wait_for_search_and_invalidates_immediately()
  {
    auto n=node(1e-8F);
    auto message=std::make_shared<se2_grid_msgs::msg::SE2Grid>();
    auto occupied=n->map_;occupied["collision_risk"][0].setConstant(1.F);
    se2_grid::SE2GridRosConverter::toMessage(occupied,*message);
    message->info.header.frame_id="map";message->info.header.stamp=n->now();
    std::unique_lock<std::mutex> held_search(n->map_mutex_);
    auto ingress=std::async(std::launch::async,[&]{n->mapCallback(message);});
    const auto status=ingress.wait_for(std::chrono::seconds(1));
    EXPECT_EQ(status,std::future_status::ready);
    if(status!=std::future_status::ready)held_search.unlock();
    ingress.get();
    const auto control=std::atomic_load(&n->control_map_snapshot_);
    ASSERT_TRUE(control);
    EXPECT_GT(n->deferred_search_maps_.load(),0u);
    double cost=0.;
    EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost,nullptr,nullptr,false,control.get()));
    EXPECT_LT(n->map_["collision_risk"][0].maxCoeff(),.5F);
    if(held_search.owns_lock())held_search.unlock();
    {
      std::lock_guard<std::mutex> lock(n->map_mutex_);n->refreshSearchMapLocked();
    }
    EXPECT_FLOAT_EQ(n->map_["collision_risk"][0].minCoeff(),1.F);
    EXPECT_FALSE(std::atomic_load(&n->pending_search_map_));
    message->info.header.frame_id="wrong_frame";
    held_search.lock();
    auto invalid=std::async(std::launch::async,[&]{n->mapCallback(message);});
    const auto invalid_status=invalid.wait_for(std::chrono::seconds(1));
    EXPECT_EQ(invalid_status,std::future_status::ready);
    held_search.unlock();invalid.get();
    EXPECT_FALSE(n->map_ready_);
    EXPECT_FALSE(std::atomic_load(&n->control_map_snapshot_));
    EXPECT_FALSE(std::atomic_load(&n->pending_search_map_));
  }
  static void provisional_motion_can_only_brake_without_changing_path_or_support()
  {
    auto n=node(.001F);n->live_obstacle_enable_=true;n->have_goal_=true;n->goal_seq_=1;
    n->live_obstacle_inflation_radius_=0.;
    n->path_={{0,0,0,0},{2,0,0,2}};n->have_path_=true;
    OdomState odom;odom.valid=true;odom.v=.35;odom.stamp=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=n->now().seconds();
    terrain_nav::MotionTrackSnapshot track;track.stamp=live->stamp_sec;track.resolution=.2;
    track.cells={{{.6,1.1}}};track.vy=-2.5;
    live->provisional_tracks={track};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    n->last_cmd_v_=.35;
    ASSERT_TRUE(n->provisionalMotionBrake(odom));
    EXPECT_EQ(n->last_cmd_v_,0.);EXPECT_EQ(n->path_.size(),2u);EXPECT_TRUE(n->have_path_);
    EXPECT_FALSE(n->dynamic_active_);EXPECT_EQ(n->provisional_brake_count_,1u);
    EXPECT_TRUE(n->provisionalMotionBrake(odom));EXPECT_EQ(n->provisional_brake_count_,1u);
    EXPECT_EQ(std::atomic_load(&n->control_map_snapshot_).get(),map.get());
    EXPECT_TRUE(live->tracks.empty());EXPECT_TRUE(live->hard.empty());
    auto away=std::make_shared<LiveObstacleSnapshot>(*live);away->provisional_tracks[0].vy=2.5;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(away));
    // The departing source at y=1.1 has no body conflict within the bounded
    // confirmation window. This warning may NOT issue a new motion command.
    EXPECT_FALSE(n->provisionalMotionBrake(odom));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    // Departure still cannot release an IMMINENT source-region conflict.
    // Retain the stopping-position hypothesis and its original uncertainty.
    away->provisional_tracks[0].cells={{{.6,.8}}};
    EXPECT_TRUE(n->provisionalMotionBrake(odom));
    away->stamp_sec-=2.;
    EXPECT_FALSE(n->provisionalMotionBrake(odom));
    n->have_goal_=false;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    EXPECT_FALSE(n->provisionalMotionBrake(odom));
  }
  static void remote_weak_motion_is_warning_not_unconditional_braking()
  {
    auto n=node(.001F);n->live_obstacle_enable_=true;n->have_goal_=true;n->goal_seq_=1;
    n->live_obstacle_inflation_radius_=0.;n->path_={{0,0,0,0},{2,0,0,2}};n->have_path_=true;
    OdomState odom;odom.valid=true;odom.v=.8;odom.stamp=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=n->now().seconds();
    terrain_nav::MotionTrackSnapshot hint;hint.stamp=live->stamp_sec;hint.resolution=.2;
    hint.cells={{{.6,3.}}};hint.vy=-.8;live->provisional_tracks={hint};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->last_cmd_v_=.8;
    EXPECT_FALSE(n->provisionalMotionBrake(odom));EXPECT_DOUBLE_EQ(n->last_cmd_v_,.8);
    EXPECT_TRUE(n->provisional_warning_active_);EXPECT_FALSE(n->provisional_brake_active_);
    EXPECT_EQ(n->path_.size(),2u);EXPECT_FALSE(n->dynamic_active_);EXPECT_TRUE(n->dynamic_unresolved_.empty());
    EXPECT_EQ(n->provisional_brake_count_,0u); // No motion or safety certificate was created.
    auto stale=std::make_shared<LiveObstacleSnapshot>(*live);stale->stamp_sec-=.4;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(stale));
    EXPECT_TRUE(n->provisionalMotionBrake(odom));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    EXPECT_FALSE(n->last_chance_accepted_);EXPECT_TRUE(n->dynamic_unresolved_.empty());
  }
  static void uncertain_first_translation_expands_only_the_braking_prediction()
  {
    auto n=node(.001F);n->live_obstacle_inflation_radius_=0.;
    LiveObstacleSnapshot live;
    terrain_nav::MotionTrackSnapshot track;track.stamp=10.;track.resolution=.2;
    track.cells={{{0.,2.}}};track.vy=-1.;live.tracks={track};
    terrain_nav::YieldState end;end.t=1.;
    const std::vector<terrain_nav::YieldState> states{terrain_nav::YieldState{},end};
    EXPECT_FALSE(n->timedMovingUnsafe(states,10.,live));
    EXPECT_TRUE(n->timedMovingUnsafe(states,10.,live,.5));
    EXPECT_FALSE(n->timedMovingUnsafe(states,10.,live));
    EXPECT_TRUE(live.hard.empty());
    EXPECT_DOUBLE_EQ(live.tracks.front().cells.front()[1],2.);
    live.tracks.front().vy=1.;
    EXPECT_FALSE(n->timedMovingUnsafe(states,10.,live,.5));
    EXPECT_TRUE(n->timedMovingUnsafe({},10.,live,.5));
    // A half-cell/scan velocity error can exceed the former allowance.
    // Only the candidate stopping prediction expands; confirmed occupancy
    // and the original data are unaffected by evaluating this envelope.
    live.tracks.front().vy=-1.;live.tracks.front().cells={{{0.,2.5}}};
    EXPECT_FALSE(n->timedMovingUnsafe(states,10.,live,.5));
    EXPECT_TRUE(n->timedMovingUnsafe(states,10.,live,n->provisional_velocity_reserve_mps_));
    EXPECT_FALSE(n->timedMovingUnsafe(states,10.,live));
    EXPECT_TRUE(live.hard.empty());
  }
  static void live_height_slice_requires_matching_geometry_and_fresh_map()
  {
    for(int mode=0;mode<9;++mode) {
      SCOPED_TRACE(mode);
      auto n=node(.001F);n->live_obstacle_min_hits_=1;n->live_obstacle_inflation_radius_=0.;
      auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      auto &grid=snapshot->grid;
      grid.add("terrain_observed",false,1.F);grid.add("gp_elevation_observed",false,0.F);
      grid.add("collision_ceiling_elevation",false,.75F);
      if(mode!=1)grid.add("collision_height_contract",false,1.F);
      grid.add("collision_height_filter_enabled",false,mode==2?0.F:1.F);
      grid.add("collision_vehicle_height_m",false,mode==3?.50F:.65F);
      grid.add("collision_headroom_margin_m",false,mode==4?.05F:.10F);
      if(mode==5)grid["collision_vehicle_height_m"][0](0,0)=.9F;
      if(mode==6)snapshot->stamp=n->now()-rclcpp::Duration::from_seconds(.31);
      if(mode==7)snapshot->stamp=n->now()+rclcpp::Duration::from_seconds(.06);
      if(mode==8) {
        grid["collision_vehicle_height_m"][0].setConstant(.8F);
        grid["collision_ceiling_elevation"][0].setConstant(.9F);
      }
      const bool dimensions_valid=mode==0 || mode==2 || mode>=6;
      EXPECT_EQ(n->heightSliceMode(grid)>=0,dimensions_valid);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
      auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
      cloud->header.frame_id="map";cloud->header.stamp=n->now();
      sensor_msgs::PointCloud2Modifier modifier(*cloud);
      modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(4);
      sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
      *x=-1.1F;*y=-1.3F;*z=1.1F;++x;++y;++z;
      *x=-.3F;*y=-1.3F;*z=.55F;++x;++y;++z;
      *x=-1.1F;*y=-1.3F;*z=0.F;++x;++y;++z;
      *x=-.3F;*y=-1.3F;*z=0.F;
      n->cloudCallback(cloud);
      const auto live=std::atomic_load(&n->live_snapshot_);ASSERT_TRUE(live);
      EXPECT_EQ(live->occupied(-1.1,-1.3),mode!=0 && mode!=8);
      EXPECT_TRUE(live->occupied(-.3,-1.3));
      if(!dimensions_valid) {
        // The normal subscriber must invalidate the old usable generation,
        // not merely ignore metadata while using already filtered occupancy.
        auto msg=std::make_shared<se2_grid_msgs::msg::SE2Grid>();
        se2_grid::SE2GridRosConverter::toMessage(grid,*msg);
        msg->info.header.frame_id="map";msg->info.header.stamp=n->now();
        n->mapCallback(msg);
        EXPECT_FALSE(n->map_ready_);
        EXPECT_FALSE(std::atomic_load(&n->control_map_snapshot_));
      }
    }
  }

  static void held_overhead_cannot_outlive_fresh_height_evidence_but_low_obstacles_do()
  {
    auto n=node(.001F);n->live_obstacle_min_hits_=1;n->live_obstacle_hold_sec_=1.;
    n->live_obstacle_ground_clear_enabled_=false;n->live_obstacle_inflation_radius_=0.;
    n->live_motion_guard_enable_=false;
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    auto &grid=snapshot->grid;
    grid.add("terrain_observed",false,1.F);grid.add("gp_elevation_observed",false,0.F);
    grid.add("collision_ceiling_elevation",false,.75F);
    grid.add("collision_height_contract",false,1.F);grid.add("collision_height_filter_enabled",false,1.F);
    grid.add("collision_vehicle_height_m",false,.65F);grid.add("collision_headroom_margin_m",false,.10F);
    auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id="map";cloud->header.stamp=n->now();
    sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(4);
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    *x=-1.1F;*y=-1.3F;*z=1.1F;++x;++y;++z;
    *x=-.3F;*y=-1.3F;*z=.55F;++x;++y;++z;
    *x=-1.1F;*y=-1.3F;*z=0.F;++x;++y;++z;
    *x=-.3F;*y=-1.3F;*z=0.F;
    snapshot->stamp=n->now()-rclcpp::Duration::from_seconds(.31);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    n->cloudCallback(cloud);
    auto live=std::atomic_load(&n->live_snapshot_);ASSERT_TRUE(live);
    EXPECT_TRUE(live->occupied(-1.1,-1.3));EXPECT_TRUE(live->occupied(-.3,-1.3));
    // Restore a fresh contract before the one-second hold expires. Its
    // height proof removes only an exclusively overhead historical reserve.
    snapshot=std::make_shared<ControlMapSnapshot>(*snapshot);snapshot->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    cloud->header.stamp=n->now();n->cloudCallback(cloud);
    live=std::atomic_load(&n->live_snapshot_);
    EXPECT_FALSE(live->occupied(-1.1,-1.3));EXPECT_TRUE(live->occupied(-.3,-1.3));
    // No new low return does not prove a low obstacle disappeared. Keep its
    // existing persistence, and preserve unknown-height reserves as well.
    modifier.resize(1);sensor_msgs::PointCloud2Iterator<float> px(*cloud,"x"),py(*cloud,"y"),pz(*cloud,"z");
    *px=-1.1F;*py=-1.3F;*pz=1.1F;n->cloudCallback(cloud);
    EXPECT_TRUE(std::atomic_load(&n->live_snapshot_)->occupied(-.3,-1.3));
  }

  static void collision_envelope_uses_width_heading_and_exactly_one_margin()
  {
    auto n=node(1e-8F);n->body_collision_front_=.58;n->body_collision_back_=.48;
    n->body_collision_half_width_=.40;n->footprint_margin_=.10;n->live_obstacle_inflation_radius_=0.;
    const auto body=n->collisionEnvelope(n->footprint_margin_);
    EXPECT_NEAR(body.front,.68,1e-9);EXPECT_NEAR(body.back,.58,1e-9);EXPECT_NEAR(body.half_width,.50,1e-9);
    LiveObstacleSnapshot live;live.min_x=live.min_y=-2;live.resolution=.02;
    live.width=live.height=200;live.inflation_radius=0.;live.hard.assign(40000,0);
    const auto obstacle=[&](double x,double y) {
      const int ix=int(std::floor((x-live.min_x)/live.resolution));
      const int iy=int(std::floor((y-live.min_y)/live.resolution));live.hard[iy*live.width+ix]=1;
    };
    obstacle(0.,.57);obstacle(0.,-.57);
    EXPECT_FALSE(n->liveBodyUnsafe(0,0,0,live));
    EXPECT_TRUE(n->liveBodyUnsafe(0,0,M_PI/2,live));
    obstacle(0.,.49);EXPECT_TRUE(n->liveBodyUnsafe(0,0,0,live));
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    std::atomic_store(&n->live_snapshot_,std::make_shared<const LiveObstacleSnapshot>(live));
    double cost=0.;EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost,nullptr,nullptr,false,snapshot.get()));
    n->chance_constrained_=false;
    EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost,nullptr,nullptr,false,snapshot.get()));
    EXPECT_TRUE(n->livePathAheadUnsafe({{0,0,0,0},{1,0,0,1}},0));
    live.hard.assign(40000,0);obstacle(0,.57);
    EXPECT_FALSE(n->liveBodyUnsafe(0,0,0,live));
    n->body_collision_half_width_=.60;
    EXPECT_TRUE(n->liveBodyUnsafe(0,0,0,live));
  }
  static void historical_obstacles_are_planning_preferences_not_safety_events()
  {
    auto n=node(1e-8F);
    const auto before=n->queryCostLocked(0,0,0);
    n->map_.add("historical_obstacle_advisory",false,1.F);n->rebuildCostmapLocked();
    const auto after=n->queryCostLocked(0,0,0);
    EXPECT_GE(after.cost,before.cost+11.9);
    EXPECT_EQ(after.probability_upper,before.probability_upper);
    EXPECT_EQ(after.known,before.known);EXPECT_EQ(after.lethal,before.lethal);
    EXPECT_EQ(after.clearance,before.clearance);
    double cost=0;EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
    const auto snapshot=std::atomic_load(&n->control_map_snapshot_);
    EXPECT_DOUBLE_EQ(n->queryCostLocked(0,0,0,false,snapshot.get()).cost,before.cost);
    terrain_nav::ChanceBudget budget;EXPECT_TRUE(n->addChancePose(0,0,0,budget,.05));
    for(auto &slice:n->map_["historical_obstacle_advisory"])slice.setZero();
    n->rebuildCostmapLocked();EXPECT_DOUBLE_EQ(n->queryCostLocked(0,0,0).cost,before.cost);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
  }
  static void primitive_clearance_matches_final_path_and_control_guard()
  {
    auto n=node(1e-8F);n->hard_clearance_floor_=.36;
    TerrainNavNode::SearchNode start;start.x=start.y=start.yaw=0.;
    ASSERT_TRUE(n->propagatePrimitiveLocked(start,0.).ok);
    std::fill(n->clearance_.begin(),n->clearance_.end(),.2F);
    double cost=0.;ASSERT_TRUE(n->footprintCostLocked(0,0,0,cost));
    EXPECT_FALSE(n->propagatePrimitiveLocked(start,0.).ok);
    std::fill(n->clearance_.begin(),n->clearance_.end(),.4F);
    EXPECT_TRUE(n->propagatePrimitiveLocked(start,0.).ok);
  }
  static void recovery_route_memory_is_soft_bounded_and_goal_scoped()
  {
    auto n=node(1e-8F);n->have_goal_=true;n->goal_seq_=1;
    const auto before=n->queryCostLocked(0,0,0);
    OdomState odom;odom.x=odom.y=0;
    n->rememberBlockedRoute(odom,1);
    const auto after=n->queryCostLocked(0,0,0);
    EXPECT_GT(after.cost,before.cost+14.);
    EXPECT_EQ(after.probability_upper,before.probability_upper);
    EXPECT_EQ(after.lethal,before.lethal);EXPECT_EQ(after.known,before.known);
    const auto snapshot=std::atomic_load(&n->control_map_snapshot_);
    EXPECT_DOUBLE_EQ(n->queryCostLocked(0,0,0,false,snapshot.get()).cost,before.cost);
    EXPECT_EQ(n->recoveryRoutePenalty(2,0),0.);
    const auto same_snapshot=std::atomic_load(&n->recovery_route_memory_);
    for(int i=0;i<10;++i)n->rememberBlockedRoute(odom,1);
    EXPECT_EQ(std::atomic_load(&n->recovery_route_memory_)->size(),1u); // same failed branch, not ten distinct ones
    EXPECT_EQ(std::atomic_load(&n->recovery_route_memory_),same_snapshot); // async work is not invalidated by retry count
    auto old_snapshot=std::make_shared<std::vector<TerrainNavNode::RecoveryRouteMemory>>(*same_snapshot);
    old_snapshot->front().stamp=n->now().seconds()-1.1;
    std::atomic_store(&n->recovery_route_memory_,std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>(old_snapshot));
    n->rememberBlockedRoute(odom,1);
    EXPECT_NE(std::atomic_load(&n->recovery_route_memory_).get(),old_snapshot.get()); // actual refresh remains enabled
    for(int i=1;i<=9;++i){auto other=odom;other.x=2.*i;n->rememberBlockedRoute(other,1);}
    EXPECT_EQ(std::atomic_load(&n->recovery_route_memory_)->size(),4u);
    EXPECT_LE(n->recoveryRoutePenalty(0,0),15.);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();
    goal->header.frame_id="map";goal->pose.position.x=3;goal->pose.orientation.w=1;
    n->goalCallback(goal);
    EXPECT_EQ(n->recoveryRoutePenalty(0,0),0.);
    n->rememberBlockedRoute(odom,1);EXPECT_EQ(n->recoveryRoutePenalty(0,0),0.);
    n->rememberBlockedRoute(odom,n->goal_seq_);
    n->goalCallback(goal);EXPECT_GT(n->recoveryRoutePenalty(0,0),14.);
    ++n->input_epoch_;EXPECT_EQ(n->recoveryRoutePenalty(0,0),0.);
  }

  static void recovery_route_memory_expires_and_never_releases_unsafe_support()
  {
    auto n=node(1e-8F);n->have_goal_=true;n->goal_seq_=1;OdomState odom;
    n->rememberBlockedRoute(odom,1);
    auto memory=std::make_shared<std::vector<TerrainNavNode::RecoveryRouteMemory>>(
        *std::atomic_load(&n->recovery_route_memory_));
    memory->front().stamp=n->now().seconds()-16.;
    std::atomic_store(&n->recovery_route_memory_,
        std::shared_ptr<const std::vector<TerrainNavNode::RecoveryRouteMemory>>(memory));
    EXPECT_EQ(n->recoveryRoutePenalty(0,0),0.);
    n->rememberBlockedRoute(odom,1);
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
    terrain_nav::ChanceBudget budget;
    EXPECT_FALSE(n->addChancePose(0,0,0,budget,.05));
  }

  static std::shared_ptr<LiveObstacleSnapshot> reserve_fixture(TerrainNavNode &n)
  {
    n.body_collision_front_=.58;n.body_collision_back_=.48;n.body_collision_half_width_=.40;
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=live->min_y=-4;live->resolution=.2;live->width=live->height=40;
    live->stamp_sec=n.now().seconds();live->inflation_radius=.25;
    live->uninflated_present.assign(1600,0);live->hard.assign(1600,0);
    live->uninflated_present[16*40+20]=1;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
      if(dx*dx+dy*dy<=1)live->hard[(16+dy)*40+20+dx]=1;
    live->present_hard=live->hard;
    std::atomic_store(&n.live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    return live;
  }

  static void reserve_recovery_keeps_physical_margin_and_exits_reserve()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    const auto live=reserve_fixture(*n);
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    ASSERT_TRUE(n->liveBodyUnsafe(0,0,0,*live));
    terrain_nav::YieldState start;terrain_nav::YieldModel model;
    const auto commands=terrain_nav::makeYieldCommands(start,model,.9,.3,0,4.5);
    ASSERT_FALSE(commands.empty());
    const auto states=terrain_nav::rolloutYield(start,start,commands,model,4.5);
    double path=0,braking=0;
    ASSERT_TRUE(n->reserveRolloutSafe(states,n->now().seconds(),commands,*map,*live,&path,&braking));
    EXPECT_LE(path,.1);EXPECT_LE(braking,.05);
    EXPECT_FALSE(n->liveBodyUnsafe(states.back().x,states.back().y,states.back().yaw,*live));
    EXPECT_EQ(live->hard[17*40+20],1);
    EXPECT_EQ(std::atomic_load(&n->live_snapshot_).get(),live.get());
    auto colliding=*live;colliding.uninflated_present[18*40+20]=1;
    EXPECT_FALSE(n->reserveRolloutSafe(states,n->now().seconds(),commands,*map,colliding));
    auto margin=*live;margin.uninflated_present[17*40+20]=1;
    EXPECT_FALSE(n->reserveRolloutSafe(states,n->now().seconds(),commands,*map,margin));
    const auto held=terrain_nav::rolloutYield(start,start,{{0,0,0}},model,4.5);
    EXPECT_FALSE(n->reserveRolloutSafe(held,n->now().seconds(),{{0,0,0}},*map,*live));
  }

  static void reserve_recovery_rejects_unknown_stale_probability_and_crossing()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    auto live=reserve_fixture(*n);const double time=n->now().seconds();
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    terrain_nav::YieldState start;terrain_nav::YieldModel model;
    const auto commands=terrain_nav::makeYieldCommands(start,model,.9,.3,0,4.5);
    const auto states=terrain_nav::rolloutYield(start,start,commands,model,4.5);
    auto unsupported=*map;
    for(auto &slice:unsupported.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,unsupported,*live));
    auto risky=*map;
    for(auto &slice:risky.grid["failure_probability_upper"])slice.setConstant(.9);
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,risky,*live));
    auto stale=*map;stale.stamp=rclcpp::Time(map->stamp.nanoseconds()-1000000000LL);
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,stale,*live));
    live->stamp_sec=time-2;
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,*map,*live));
    live->stamp_sec=time;
    terrain_nav::MotionTrackSnapshot track;track.cells={{{-.4,2}}};track.vy=-1;
    track.stamp=time;track.resolution=.1;live->tracks={track};
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,*map,*live));
    live->tracks.clear();live->uninflated_present.clear();
    EXPECT_FALSE(n->reserveRolloutSafe(states,time,commands,*map,*live));
  }

  static void reserve_recovery_control_is_bounded_and_cancelled()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
    const auto live=reserve_fixture(*n);
    ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
    EXPECT_TRUE(n->reserve_active_);EXPECT_LT(n->last_cmd_v_,0);EXPECT_GE(n->last_cmd_v_,-.30);
    EXPECT_EQ(n->reverse_recovery_count_,0);
    EXPECT_EQ(n->reserve_recovery_count_,1);
    EXPECT_GE(n->reserve_forward_exit_.size(),2u);EXPECT_LE(n->reserve_joint_upper_,.1);
    ++n->goal_seq_;
    EXPECT_TRUE(n->reserveRecoveryControl(n->odom_));
    EXPECT_FALSE(n->reserve_active_);EXPECT_EQ(n->last_cmd_v_,0);
    n->reserve_last_attempt_=-1e9;n->reserve_recovery_count_=2;
    EXPECT_FALSE(n->reserveRecoveryControl(n->odom_));
    n->reserve_recovery_count_=0;n->odom_.v=.2;
    EXPECT_FALSE(n->reserveRecoveryControl(n->odom_));
    n->odom_.v=0;n->reverse_recovery_enable_=false;
    EXPECT_FALSE(n->reserveRecoveryControl(n->odom_));
  }

  static void reserve_and_maneuver_attempts_are_independent_and_reset_for_new_goal()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
    reserve_fixture(*n);n->reverse_recovery_count_=2;
    ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
    EXPECT_EQ(n->reverse_recovery_count_,2);EXPECT_EQ(n->reserve_recovery_count_,1);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();
    goal->header.frame_id="map";goal->pose.position.x=4.;goal->pose.orientation.w=1.;
    n->goalCallback(goal);
    EXPECT_EQ(n->reverse_recovery_count_,0);EXPECT_EQ(n->reserve_recovery_count_,0);
    const auto seq=n->goal_seq_;
    n->reverse_recovery_count_=2;n->reserve_recovery_count_=2;
    n->goalCallback(goal);
    EXPECT_EQ(n->goal_seq_,seq);
    EXPECT_EQ(n->reverse_recovery_count_,2);EXPECT_EQ(n->reserve_recovery_count_,2);
  }

  static void reserve_recovery_completes_only_after_stopped_fresh_release()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
    const auto live=reserve_fixture(*n);
    ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
    n->odom_.x=-.9;n->odom_.v=0;n->odom_.stamp=n->now();
    n->reserve_origin_=n->now().seconds()-6.5+5e-8;
    n->reserve_release_since_=n->now().seconds()-.3;
    n->reserve_release_stamp_=std::atomic_load(&n->control_map_snapshot_)->stamp.seconds()-.1;
    EXPECT_TRUE(n->reserveRecoveryControl(n->odom_));
    EXPECT_FALSE(n->reserve_active_);EXPECT_EQ(n->reserve_state_,"completed");
    EXPECT_EQ(n->last_cmd_v_,0);
  }

  static void reserve_forward_exit_requires_progress_and_shared_budget()
  {
    auto n=node(1e-8F);n->footprint_margin_=.1;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
    const auto live=reserve_fixture(*n);
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    terrain_nav::YieldState start;terrain_nav::YieldModel model;
    ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
    const auto retreat=terrain_nav::rolloutYield(start,start,n->reserve_commands_,model,4.5);
    std::vector<Pose2> exit;double upper=0;
    ASSERT_TRUE(n->reserveForwardExit(retreat,{3,0,0,0},*map,*live,exit,&upper));
    EXPECT_GE(exit.back().x-start.x,.25);EXPECT_LE(upper,.1);
    for(const auto &p:exit)EXPECT_FALSE(n->liveBodyUnsafe(p.x,p.y,p.yaw,*live));
    EXPECT_TRUE(n->bodyArcPathFeasible(exit));
    auto blocked=*live;std::fill(blocked.hard.begin(),blocked.hard.end(),1);
    EXPECT_FALSE(n->reserveForwardExit(retreat,{3,0,0,0},*map,blocked,exit));
    EXPECT_TRUE(exit.empty());
    auto unknown=*map;
    for(auto &slice:unknown.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->reserveForwardExit(retreat,{3,0,0,0},unknown,*live,exit));
    EXPECT_FALSE(n->reserveForwardExit(retreat,{0,0,0,0},*map,*live,exit));
  }

  static void reserve_exit_handoff_recertifies_actual_pose_and_blocks_old_planner()
  {
    for(bool invalidate:{false,true}) {
      auto n=node(1e-8F);n->footprint_margin_=.1;
      n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
      n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
      reserve_fixture(*n);
      ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
      EXPECT_FALSE(n->planSnapshotCurrent(1,n->input_epoch_.load()));
      EXPECT_FALSE(n->planPath("old planner cannot replace active retreat"));
      const auto endpoint=n->reserve_forward_exit_.front();
      n->odom_.x=endpoint.x;n->odom_.y=endpoint.y;n->odom_.yaw=endpoint.yaw;n->odom_.v=0;
      n->reserve_origin_=n->now().seconds()-6.5+5e-8;
      n->reserve_release_since_=n->now().seconds()-.3;
      n->reserve_release_stamp_=std::atomic_load(&n->control_map_snapshot_)->stamp.seconds()-.1;
      ASSERT_TRUE(n->reserveRecoveryControl(n->odom_));
      ASSERT_FALSE(n->reserve_retreat_active_);ASSERT_TRUE(std::atomic_load(&n->reserve_exit_request_));
      EXPECT_FALSE(n->have_path_);EXPECT_EQ(n->last_cmd_v_,0);
      if(invalidate)for(auto &slice:n->map_["probability_valid"])slice.setZero();
      const bool planned=n->planPath("reserve exit handoff qualification");
      if(invalidate) {EXPECT_FALSE(planned);EXPECT_FALSE(n->have_path_);}
      else {
        ASSERT_TRUE(planned);EXPECT_TRUE(n->recovery_path_);EXPECT_TRUE(n->recovery_exit_active_);
        EXPECT_NEAR(n->path_.front().x,n->odom_.x,1e-9);
        EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
      }
    }
  }

  static void side_obstacle_can_trigger_certified_recovery_but_not_unknown_motion()
  {
    auto n=node(1e-8F);n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->live_obstacle_enable_=true;n->have_goal_=true;n->goal_seq_=1;
    n->raw_goal_={3,0,0,0};n->odom_.valid=true;n->odom_.stamp=n->now();
    auto live=reserve_fixture(*n);
    std::fill(live->hard.begin(),live->hard.end(),0);
    live->hard[25*40+20]=1;live->present_hard=live->uninflated_present=live->hard;
    for(double s=.6;s<2.5;s+=.1)for(double y:{-.35,0.,.35})
      ASSERT_FALSE(live->occupied(s,y));
    ASSERT_TRUE(n->planReverseRecovery());
    EXPECT_TRUE(n->recovery_path_);EXPECT_EQ(n->reverse_recovery_count_,1);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    EXPECT_FALSE(n->recovery_exit_.empty());
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->planReverseRecovery());
    EXPECT_EQ(n->reverse_recovery_count_,1);
  }

  static void same_xy_ground_does_not_erase_a_thin_upper_obstacle()
  {
    auto n=node(.001F);n->live_obstacle_min_hits_=1;
    EXPECT_FALSE(n->live_obstacle_ground_clear_enabled_);
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    snapshot->grid.add("terrain_observed",false,1.F);
    snapshot->grid.add("gp_elevation_observed",false,0.F);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id="map";cloud->header.stamp=n->now();
    sensor_msgs::PointCloud2Modifier modifier(*cloud);modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(1);
    sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
    *x=-1.1F;*y=-1.3F;*z=.6F;n->cloudCallback(cloud);
    ASSERT_TRUE(std::atomic_load(&n->live_snapshot_)->occupied(-1.1,-1.3));
    *z=0;
    n->cloudCallback(cloud);n->cloudCallback(cloud);
    EXPECT_TRUE(std::atomic_load(&n->live_snapshot_)->occupied(-1.1,-1.3));
    for(auto &expiry:n->live_expiry_)expiry=n->now().seconds()-.01;
    n->cloudCallback(cloud);
    EXPECT_FALSE(std::atomic_load(&n->live_snapshot_)->occupied(-1.1,-1.3));
  }

  static void braking_envelope_keeps_momentum_during_gear_change()
  {
    for(double direction:{-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      OdomState odom;odom.valid=true;odom.v=direction;
      geometry_msgs::msg::Twist command;command.linear.x=-direction*.3;
      EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,command));
      Eigen::Array3i obstacle;
      ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(direction*1.1,.1,0),obstacle));
      n->map_.at("collision_risk",obstacle)=1;
      n->rebuildCostmapLocked();
      auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      snapshot->grid=n->map_;
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
      double cost=0;
      ASSERT_TRUE(n->footprintCostLocked(0,0,0,cost));
      EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
      command.linear.x=0;
      EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
      // At rest the same free reverse corridor is admissible; neither a
      // moving safe bubble nor a blanket prohibition on reversing is added.
      odom.v=0;command.linear.x=-direction*.3;
      EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,command));
    }
  }

  static void time_indexed_final_command_rejects_each_actor_and_keeps_current_occupancy()
  {
    auto n=node(1e-8F);n->dynamic_time_indexed_routing_=true;n->live_obstacle_enable_=true;
    n->live_obstacle_inflation_radius_=0.;n->dynamic_yield_deceleration_=1.2;
    const double time=n->now().seconds();OdomState odom;odom.valid=true;odom.stamp=n->now();odom.v=.3;
    geometry_msgs::msg::Twist command;command.linear.x=.3;
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4;
    live->width=live->height=40;live->resolution=.2;live->stamp_sec=time;
    live->hard.assign(1600,0);live->present_hard=live->hard;live->forecast.assign(1600,1);
    terrain_nav::MotionTrackSnapshot departed;departed.id=1;departed.resolution=.1;
    departed.stamp=time;departed.cells={{{-2.,-3.}}};departed.vy=-1.5;live->tracks={departed};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,command));
    // The second actor intersects a FUTURE stop, not the current pose.
    auto crossing=departed;crossing.id=2;crossing.cells={{{.65,3.}}};crossing.vy=-1.5;
    live->tracks.push_back(crossing);
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
    live->tracks={departed};live->hard.assign(1600,1);live->present_hard=live->hard;
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
    live->hard.assign(1600,0);live->present_hard=live->hard;
    auto unknown=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    for(auto &slice:unknown->grid["probability_valid"])slice.setZero();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(unknown));
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
    live->stamp_sec-=1.;EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,command));
  }

  static void measured_cloud_routing_never_turns_future_sweep_into_permanent_geometry()
  {
    for(bool indexed:{false,true}) {
      auto n=node(1e-8F);n->dynamic_time_indexed_routing_=indexed;
      n->live_obstacle_resolution_=.1;n->live_motion_min_resolution_=.1;
      n->live_motion_horizon_sec_=1.5;n->live_obstacle_min_hits_=1;
      auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      map->grid.add("terrain_observed",false,1.F);map->grid.add("gp_elevation_observed",false,0.F);
      map->stamp=rclcpp::Time(100000000000LL,RCL_ROS_TIME);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      for(int k=0;k<4;++k) {
        ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL+k*100000000LL),RCL_RET_OK);
        auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
        cloud->header.frame_id="map";cloud->header.stamp=n->now();
        sensor_msgs::PointCloud2Modifier m(*cloud);m.setPointCloud2FieldsByString(1,"xyz");m.resize(9);
        sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
        for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b,++x,++y,++z) {
          *x=.65F+a*.1F;*y=2.65F-k*.2F+b*.1F;*z=.6F;
        }
        n->cloudCallback(cloud);
      }
      const auto live=std::atomic_load(&n->live_snapshot_);ASSERT_TRUE(live);
      ASSERT_FALSE(live->tracks.empty());size_t future_only=0;
      for(size_t i=0;i<live->forecast.size();++i)if(live->forecast[i]&&!live->present_hard[i]) {
        ++future_only;EXPECT_EQ(bool(live->hard[i]),!indexed);
      }
      EXPECT_GT(future_only,0u);
      for(size_t i=0;i<live->present_hard.size();++i)if(live->present_hard[i])EXPECT_TRUE(live->hard[i]);
    }
  }

  static void directional_braking_distinguishes_parallel_side_walls_from_turning_into_them()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->wheel_base_=.430;n->rear_axle_offset_=.215;n->max_steer_=.3143291323113041;
    n->body_collision_front_=.421;n->body_collision_back_=.421;n->body_collision_half_width_=.360;
    n->footprint_margin_=.1;n->hard_clearance_floor_=0.; // Actual NEOS YAML: one body margin, no second EDT reserve.
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    for(int r=0;r<n->map_.getSizePos()[0];++r)for(int c=0;c<n->map_.getSizePos()[1];++c) {
      Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
      if(std::abs(std::abs(p.y())-.7)<.05)n->map_.at("collision_risk",Eigen::Array3i(r,c,0))=1;
    }
    n->rebuildCostmapLocked();
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->grid=n->map_;map->costs=n->costs_;map->clearance=n->clearance_;map->comfort_clearance=n->comfort_clearance_;map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    OdomState odom;odom.valid=true;odom.v=1.;
    geometry_msgs::msg::Twist cmd;cmd.linear.x=1.;double clearance=0,cost=0,upper=1;
    ASSERT_TRUE(n->footprintCostLocked(0,0,0,cost,&clearance,nullptr,false,map.get()));
    EXPECT_LT(terrain_nav::clearanceSpeedLimit(clearance,n->hardClearanceFloor(),.45,1.2),.7);
    EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,cmd,&upper));
    EXPECT_LT(upper,n->chance_braking_budget_);
    for(double sign:{-1.,1.}) {
      cmd.angular.z=sign*n->max_steer_;
      EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
    }
    cmd.angular.z=0;cmd.linear.x=-1.;odom.v=-1.;
    EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,cmd));
    // Unknown ground on an otherwise empty stopping corridor is not free.
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
  }

  static void directional_braking_retains_source_latency_and_uses_the_weaker_deceleration()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;n->max_accel_=2.;n->dynamic_yield_deceleration_=1.2;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    Eigen::Array3i obstacle;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.7,.1,0),obstacle));
    n->map_.at("collision_risk",obstacle)=1;n->rebuildCostmapLocked();
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->grid=n->map_;map->costs=n->costs_;map->clearance=n->clearance_;map->comfort_clearance=n->comfort_clearance_;map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    OdomState odom;odom.valid=true;odom.v=1.;geometry_msgs::msg::Twist cmd;cmd.linear.x=1.;
    EXPECT_TRUE(n->commandBrakingEnvelopeSafe(odom,cmd));
    map->stamp=map->stamp-rclcpp::Duration::from_seconds(.4);
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
    map->stamp=n->now();n->dynamic_yield_deceleration_=.3;
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(odom,cmd));
  }

  static void observation_controller_caps_only_observation_generation()
  {
    for(bool observation:{true,false}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=.8;
      n->have_goal_=true;n->raw_goal_={3,0,0,0};
      n->have_path_=true;n->path_={{0,0,0,0},{3,0,0,3}};
      n->path_seq_=10;n->observation_path_seq_=observation ? 10 : 9;
      n->last_cmd_v_=.8;n->last_cmd_time_sec_=n->now().seconds();
      n->controlTimer();
      if(observation)EXPECT_LE(std::abs(n->last_cmd_v_),.35);
      else EXPECT_GT(n->last_cmd_v_,.35);
    }
  }
  static void dynamic_yield_certifies_time_not_union(bool input_recovery=false)
  {
    auto n=node(1e-8F);
    n->footprint_front_=.42;n->body_collision_front_=.58;
    n->footprint_back_=.42;n->body_collision_back_=.48;
    n->footprint_half_width_=.31;n->body_collision_half_width_=.4;
    n->footprint_margin_=.1;n->live_obstacle_inflation_radius_=.05;
    const double time=n->now().seconds();
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=-4;live->min_y=-4;live->resolution=.2;live->width=40;live->height=40;
    live->stamp_sec=time;live->hard.assign(1600,1);live->present_hard.assign(1600,0);
    terrain_nav::MotionTrackSnapshot track;
    track.cells={{{.65,3.}}};track.vy=-1.5;track.stamp=time;track.resolution=.1;
    live->tracks={track};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    terrain_nav::YieldState start;terrain_nav::YieldModel model;
    const std::vector<terrain_nav::YieldCommand> brake{{0,0,0}};
    const auto held=terrain_nav::rolloutYield(start,start,brake,model,3);
    EXPECT_TRUE(n->timedMovingUnsafe(held,time,*live));
    std::string failure;
    EXPECT_FALSE(n->timedYieldSafe(held,time,brake,*map,*live,nullptr,nullptr,&failure));
    EXPECT_EQ(failure,"moving obstacle reservation");
    const auto commands=terrain_nav::makeYieldCommands(start,model,.6,.55,0,3);
    ASSERT_FALSE(commands.empty());
    const auto retreat=terrain_nav::rolloutYield(start,start,commands,model,3);
    EXPECT_TRUE(n->timedYieldSafe(retreat,time,commands,*map,*live,nullptr,nullptr,&failure));
    EXPECT_EQ(failure,"certified");
    EXPECT_EQ(std::atomic_load(&n->live_snapshot_).get(),live.get());
    EXPECT_EQ(live->hard.front(),1);
    EXPECT_NEAR(retreat.back().speed,0,1e-9);
    EXPECT_LT(retreat.back().x,-.4);
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->live_obstacle_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};n->odom_.valid=true;n->odom_.stamp=n->now();
    if(input_recovery) {
      n->stale_inputs_waiting_=true;
      EXPECT_FALSE(n->dynamicYieldControl(n->odom_)); // Nominal debounce still applies.
      n->controlTimer(); // The REAL timer may execute a newly certified safety retreat.
      EXPECT_TRUE(n->stale_inputs_waiting_); // Does not resume the ordinary path.
    } else ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_TRUE(n->dynamic_active_);EXPECT_LT(n->last_cmd_v_,0);
    EXPECT_EQ(n->dynamic_forward_attempts_,0u); // A forward goal does not force an acceleration race.
    EXPECT_TRUE(std::isfinite(n->dynamic_path_upper_));
    EXPECT_TRUE(std::isfinite(n->dynamic_braking_upper_));
    if(input_recovery) {
      const auto fresh_map=std::atomic_load(&n->control_map_snapshot_);
      auto stale=std::make_shared<ControlMapSnapshot>(*fresh_map);
      stale->stamp=stale->stamp-rclcpp::Duration::from_seconds(2.);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(stale));
      n->controlTimer();EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      EXPECT_TRUE(n->stale_inputs_waiting_);EXPECT_FALSE(n->dynamic_active_);
      std::atomic_store(&n->control_map_snapshot_,fresh_map);
    }
    auto second=track;second.cells={{{-.65,-3.}}};second.vy=1.5;
    live->tracks.push_back(second);
    EXPECT_FALSE(n->timedYieldSafe(retreat,time,commands,*map,*live,nullptr,nullptr,&failure));
    EXPECT_EQ(failure,"moving obstacle reservation");
    live->tracks.resize(1);
    for(int iy=0;iy<40;++iy)for(int ix=0;ix<40;++ix)
      live->present_hard[iy*40+ix]=(-4+(ix+.5)*.2)<-1.;
    EXPECT_FALSE(n->timedYieldSafe(retreat,time,commands,*map,*live));
    live->present_hard.assign(1600,0);
    auto unsupported=*map;
    for(int r=0;r<40;++r)for(int c=0;c<40;++c) {
      Eigen::Vector3d p;unsupported.grid.index2Pos(Eigen::Array3i(r,c,0),p);
      if(p.x()<-.2)for(auto &slice:unsupported.grid["probability_valid"])slice(r,c)=0;
    }
    EXPECT_FALSE(n->timedYieldSafe(retreat,time,commands,unsupported,*live));
    live->tracks[0].stamp=time-1.;
    EXPECT_FALSE(n->timedYieldSafe(retreat,time,commands,*map,*live));
  }

  static void timed_yield_shared_ledger_matches_independent_reference()
  {
    for(float probability:{.001F,.004F}) {
      auto n=node(probability);n->live_obstacle_enable_=true;
      auto *clock=n->get_clock()->get_clock_handle();
      ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
      LiveObstacleSnapshot live;live.min_x=live.min_y=-4.;live.resolution=.2;live.width=live.height=40;
      live.stamp_sec=100.;live.hard.assign(1600,0);live.present_hard=live.hard;
      terrain_nav::YieldState start;start.speed=.3;
      terrain_nav::YieldModel model;
      for(int direction:{-1,1})for(double steering:{-.25,0.,.25}) {
        const auto commands=direction>0?terrain_nav::makeAdvanceCommands(start,model,1.2,.55,steering,4.):
          terrain_nav::makeYieldCommands(start,model,.8,.4,steering,4.);
        ASSERT_FALSE(commands.empty());
        const auto states=terrain_nav::rolloutYield(start,start,commands,model,4.);
        ASSERT_NEAR(states.back().speed,0.,.001);
        for(auto limits:std::vector<std::pair<double,double>>{{.05,.10},{.10,.01},{.01,.10}}) {
          n->chance_path_budget_=limits.first;n->chance_braking_budget_=limits.second;
          terrain_nav::ChanceBudget reference_path,reference_brake;bool expected=true;
          std::string expected_failure;
          for(const auto &s:states) {
            if(!n->addChancePose(s.x,s.y,s.yaw,reference_path,limits.first,map.get(),&live)) {
              expected=false;expected_failure="path probability budget";break;
            }
            const double request=terrain_nav::yieldCommandAt(commands,s.t).speed;
            if((request==0 || request*s.speed<0 || std::abs(request)<std::abs(s.speed)-1e-8) &&
                !n->addChancePose(s.x,s.y,s.yaw,reference_brake,limits.second,map.get(),&live)) {
              expected=false;expected_failure="braking probability budget";break;
            }
          }
          terrain_nav::ChanceBudget reference_current_stop;
          if(expected) {
            auto stopping_model=model;stopping_model.delay=.45;
            stopping_model.acceleration=n->max_accel_;
            const auto stopped=terrain_nav::rolloutYield(start,start,{{0,0,0}},stopping_model,
                .45+std::abs(start.speed)/1.2+.10);
            for(const auto &s:stopped)
              if(!n->addChancePose(s.x,s.y,s.yaw,reference_current_stop,limits.second,map.get(),&live)) {
                expected=false;expected_failure="current stopping probability budget";break;
              }
          }
          double path,brake;std::string failure;
          EXPECT_EQ(n->timedYieldSafe(states,100.,commands,*map,live,&path,&brake,&failure),expected);
          if(expected) {
            EXPECT_DOUBLE_EQ(path,reference_path.upper());
            EXPECT_DOUBLE_EQ(brake,std::max(reference_brake.upper(),reference_current_stop.upper()));
          } else EXPECT_EQ(failure,expected_failure);
        }
      }
    }
  }

  static void forward_cruising_is_not_the_whole_braking_policy()
  {
    auto n=node(.004F);n->live_obstacle_enable_=true;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
    LiveObstacleSnapshot live;live.min_x=live.min_y=-4.;live.resolution=.2;live.width=live.height=40;
    live.stamp_sec=100.;live.hard.assign(1600,0);live.present_hard=live.hard;
    terrain_nav::YieldState start;start.speed=.3;terrain_nav::YieldModel model;
    const auto commands=terrain_nav::makeAdvanceCommands(start,model,3.,.8,0.,6.);
    ASSERT_FALSE(commands.empty());const auto states=terrain_nav::rolloutYield(start,start,commands,model,6.);
    double path=0,brake=0;std::string failure;
    ASSERT_TRUE(n->timedYieldSafe(states,100.,commands,*map,live,&path,&brake,&failure));
    EXPECT_GT(path,.05);EXPECT_LE(path,.10);EXPECT_LE(brake,.05);
    // The same complete route still fails its ORIGINAL path limit, even
    // though its separately evaluated stopping policies are below 5%.
    n->chance_path_budget_=.05;
    EXPECT_FALSE(n->timedYieldSafe(states,100.,commands,*map,live,nullptr,nullptr,&failure));
    EXPECT_EQ(failure,"path probability budget");
    n->chance_path_budget_=.10;n->chance_braking_budget_=.001;
    EXPECT_FALSE(n->timedYieldSafe(states,100.,commands,*map,live));
    // A missing current stopping footprint never becomes permissive because
    // the selected motion has a certified terminal stop.
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(states,100.,commands,*map,live));
  }

  static void safety_root_is_current_while_tracking_lead_keeps_original_source_clock()
  {
    auto n=node(1e-8F);auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    OdomState raw;raw.valid=true;raw.stamp=rclcpp::Time(99900000000LL);raw.v=1.;raw.yaw_rate=0.;
    n->odom_pose_prediction_time_=.08;
    auto safety=raw,tracking=raw;
    n->predictOdomForControl(safety,false);n->predictOdomForControl(tracking);
    EXPECT_NEAR(safety.x,.10,1e-8);EXPECT_NEAR(tracking.x,.18,1e-8);
    EXPECT_EQ(safety.stamp.nanoseconds(),raw.stamp.nanoseconds());
    EXPECT_EQ(tracking.stamp.nanoseconds(),raw.stamp.nanoseconds());
    EXPECT_NEAR(n->now().seconds()-safety.stamp.seconds(),.1,1e-8);
  }

  static void control_root_uses_the_same_separate_actuator_delays_as_maneuver_proofs()
  {
    auto n=node(1e-8F);auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->dynamic_speed_dead_time_=0.;n->dynamic_steering_dead_time_=.06;
    n->dynamic_yield_delay_=.2;n->dynamic_yield_deceleration_=n->max_accel_=1.2;
    n->wheel_base_=.43;n->rear_axle_offset_=.215;n->max_steer_rate_=4.;
    const std::vector<terrain_nav::YieldCommand> issued{{99.7,1.,0.},{99.83,1.,.2},{99.94,.5,.2}};
    for(auto c:issued)ASSERT_TRUE(n->issued_commands_.record(c));
    OdomState measured;measured.valid=true;measured.stamp=rclcpp::Time(99800000000LL);measured.v=1.;
    auto actual=measured;n->predictOdomForControl(actual,false);ASSERT_TRUE(actual.valid);
    auto model=n->configuredYieldModel();model.step=.005;
    const auto expected=terrain_nav::rolloutQueuedYield({0.,0.,0.,0.,1.,0.},
      {{-.06,1.,0.},{.03,1.,.2},{.14,.5,.2}},model,.2,99.8,issued);
    ASSERT_FALSE(expected.empty());EXPECT_NEAR(actual.x,expected.back().x,1e-10);
    EXPECT_NEAR(actual.y,expected.back().y,1e-10);EXPECT_NEAR(actual.yaw,expected.back().yaw,1e-10);
    EXPECT_NEAR(actual.v,expected.back().speed,1e-10);EXPECT_EQ(actual.stamp,measured.stamp);
    EXPECT_GT(actual.yaw,.01);EXPECT_LT(actual.v,1.);
  }

  static void control_root_replays_actual_interframe_inputs_without_relabeling_source_measurements()
  {
    auto n=node(1e-8F);auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->dynamic_yield_delay_=.04;n->dynamic_yield_deceleration_=1.2;n->max_accel_=1.2;
    n->wheel_base_=.43;n->rear_axle_offset_=.215;n->max_steer_=.31432913231137;
    n->max_steer_rate_=4.;n->odom_pose_prediction_time_=.08;
    OdomState raw;raw.valid=true;raw.stamp=rclcpp::Time(99800000000LL);
    ASSERT_TRUE(n->issued_commands_.record({99.,0.,0.}));
    ASSERT_TRUE(n->issued_commands_.record({99.76,1.,0.}));
    auto predicted=raw;n->predictOdomForControl(predicted,false);
    ASSERT_TRUE(predicted.valid);EXPECT_NEAR(predicted.x,.024,1e-8);EXPECT_NEAR(predicted.v,.24,1e-8);
    EXPECT_TRUE(predicted.command_prediction);
    EXPECT_DOUBLE_EQ(predicted.prediction_source[3],0.);EXPECT_EQ(predicted.stamp,raw.stamp);
    EXPECT_NEAR(predicted.prediction_sec,.2,1e-8);
    n->predictOdomForControl(predicted,false);EXPECT_NEAR(predicted.x,.024,1e-8); // Not integrated twice.
    n->predictOdomForControl(predicted,true);EXPECT_NEAR(predicted.x,.04704,1e-8); // .20s age plus .08s lead.
    n->predictOdomForControl(predicted,false);EXPECT_NEAR(predicted.x,.024,1e-8); // Lead cannot leak into safety.
    raw.v=.8;raw.yaw_rate=.8*std::tan(.2)/.43;
    n->issued_commands_=terrain_nav::IssuedYieldHistory{};
    ASSERT_TRUE(n->issued_commands_.record({99.,.8,.2}));
    ASSERT_TRUE(n->issued_commands_.record({99.76,.8,-.2}));
    auto turn=raw;n->predictOdomForControl(turn,false);ASSERT_TRUE(turn.valid);
    const auto constant=terrain_nav::integrateBodyVelocity(raw,raw.v,raw.yaw_rate,.2,.215);
    EXPECT_LT(turn.yaw,0.);EXPECT_GT(constant.yaw,0.);EXPECT_GT(constant.yaw-turn.yaw,.10);
    EXPECT_NEAR(turn.yaw_rate,-.8*std::tan(.2)/.43,1e-8);
    EXPECT_DOUBLE_EQ(turn.prediction_source[4],raw.yaw_rate);
    EXPECT_EQ(turn.stamp,raw.stamp);
    auto slip=raw;slip.v=.1;slip.yaw_rate=.8;
    const auto measured=terrain_nav::integrateBodyVelocity(slip,slip.v,slip.yaw_rate,.2,.215);
    n->predictOdomForControl(slip,false);
    EXPECT_FALSE(slip.command_prediction);EXPECT_NEAR(slip.yaw,measured.yaw,1e-8);
    EXPECT_NEAR(slip.x,measured.x,1e-8);EXPECT_DOUBLE_EQ(slip.v,.1); // Do not erase measured spin/slip.
    auto delayed=raw;delayed.stamp=rclcpp::Time(99650000000LL);
    n->predictOdomForControl(delayed,false);
    ASSERT_TRUE(delayed.valid);EXPECT_NEAR(delayed.prediction_sec,.35,1e-8);
    EXPECT_NEAR(delayed.stamp.seconds()+delayed.prediction_sec,n->now().seconds(),1e-8);
    auto expired=raw;expired.stamp=rclcpp::Time(99000000000LL);
    n->predictOdomForControl(expired,false);EXPECT_DOUBLE_EQ(expired.prediction_sec,0.);
    EXPECT_DOUBLE_EQ(expired.x,raw.x); // Remains stale; the ordinary source-age gate still rejects it.
    auto future=raw;future.stamp=rclcpp::Time(101000000000LL);
    n->predictOdomForControl(future,false);EXPECT_FALSE(future.valid);
    ASSERT_TRUE(n->issued_commands_.record({101.,0.,0.}));
    auto old_epoch=raw;n->predictOdomForControl(old_epoch,false);EXPECT_FALSE(old_epoch.valid);
  }

  static void fresh_nominal_motion_can_handover_without_hypothetical_old_steering_lock(bool paused=false)
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->hard_clearance_floor_=0.;n->terrain_speed_enable_=false;
    n->reverse_recovery_enable_=true;n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};
    n->odom_.valid=true;n->odom_.v=1.2;n->odom_.yaw_rate=1.2*std::tan(.4)/n->wheel_base_;
    const double time=n->now().seconds();n->odom_.stamp=n->now();
    n->have_path_=true;n->path_={{0,0,0,0},{1,0,0,1},{2,0,0,2},{3,0,0,3}};
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=live->min_y=-4.;live->width=live->height=40;live->resolution=.2;
    live->stamp_sec=time;live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot track;track.id=7;track.stamp=time;track.resolution=.1;
    track.cells={{{-1.3,1.2}}};track.vy=-.05;live->tracks={track};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto map=std::atomic_load(&n->control_map_snapshot_);
    terrain_nav::YieldState state;state.speed=1.2;state.steering=.4;
    terrain_nav::YieldModel model;model.wheelbase=n->wheel_base_;
    const auto old_held=terrain_nav::rolloutYield(state,state,{{0,1.2,.4}},model,5.5);
    ASSERT_TRUE(n->timedMovingUnsafe(old_held,time,*live));
    ASSERT_TRUE(n->nominalDynamicHandoverSafe(n->odom_,state,model,time,5.5,*map,*live));
    if(paused)n->have_path_=false;
    n->dynamic_active_=true;n->dynamic_origin_=time;n->dynamic_last_time_=time;
    n->dynamic_goal_seq_=1;n->dynamic_epoch_=n->input_epoch_.load();n->dynamic_horizon_sec_=5.5;
    n->dynamic_conflict_goal_=1;n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
    n->dynamic_unresolved_[7]=time;
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_,paused));
    EXPECT_FALSE(n->dynamic_active_);EXPECT_TRUE(n->dynamic_unresolved_.empty());
    // The next tick must not re-latch solely because that same OLD steering
    // would trace a hypothetical loop; final normal control still checks it.
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_,paused));EXPECT_TRUE(n->dynamic_unresolved_.empty());
    if(paused) {
      EXPECT_EQ(n->last_cmd_v_,0.); // Ending interaction authorizes no new motion.
      // Lost evidence before a fresh separation proof is still a hard hold.
      n->dynamic_unresolved_[7]=time;live->tracks.clear();
      EXPECT_TRUE(n->dynamicYieldControl(n->odom_,true));
      EXPECT_EQ(n->dynamic_yield_state_,"prediction_lost_hold");
      EXPECT_TRUE(n->dynamic_unresolved_.count(7));
      return;
    }
    live->present_hard.assign(1600,1);
    EXPECT_FALSE(n->nominalDynamicHandoverSafe(n->odom_,state,model,time,5.5,*map,*live));
    live->present_hard.assign(1600,0);
    auto unknown=*map;for(auto &slice:unknown.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->nominalDynamicHandoverSafe(n->odom_,state,model,time,5.5,unknown,*live));
    n->recovery_path_=true;
    EXPECT_FALSE(n->nominalDynamicHandoverSafe(n->odom_,state,model,time,5.5,*map,*live));
  }

  static void early_braking_can_stop_in_a_crossing_despite_a_clear_nominal_endpoint()
  {
    auto n=node(1e-8F);n->body_collision_front_=.421;n->body_collision_back_=.421;
    n->body_collision_half_width_=.36;n->footprint_front_=.42;n->footprint_back_=.42;
    n->footprint_half_width_=.31;n->footprint_margin_=.1;n->live_obstacle_inflation_radius_=0.;
    terrain_nav::YieldState start;start.speed=1.2;
    const auto braking=terrain_nav::rolloutYield(start,start,{{0,0,0}},{},4.);
    LiveObstacleSnapshot live;terrain_nav::MotionTrackSnapshot track;
    track.cells={{{-.2,2.}}};track.vy=-1.;track.resolution=.1;track.stamp=100.;live.tracks={track};
    EXPECT_FALSE(n->timedMovingUnsafe(braking,100.,live));
    const auto alternatives=terrain_nav::possibleEarlyStops(braking);
    EXPECT_TRUE(n->timedMovingUnsafe(alternatives,100.,live));
    EXPECT_EQ(alternatives.size(),braking.size()+1);
    EXPECT_DOUBLE_EQ(alternatives.front().t,braking.front().t);
    for(size_t i=1;i<alternatives.size();++i) {
      EXPECT_DOUBLE_EQ(alternatives[i].x,braking[i-1].x);
      EXPECT_DOUBLE_EQ(alternatives[i].t,braking.back().t);
    }
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=rclcpp::Time(100000000000LL,RCL_ROS_TIME);
    live.width=live.height=40;live.min_x=live.min_y=-4.;live.resolution=.2;live.stamp_sec=100.;
    live.hard.assign(1600,0);live.present_hard=live.hard;
    std::string failure;
    EXPECT_FALSE(n->timedYieldSafe(braking,100.,{{0,0,0}},*map,live,nullptr,nullptr,&failure));
    EXPECT_EQ(failure,"possible earlier stopping position");
  }

  static void dynamic_handover_checks_the_short_reverse_leg_and_releases_it_to_tracking()
  {
    auto n=node(1e-8F);
    n->motion_prediction_model_="gaussian_joint_cpu";
    n->live_obstacle_enable_=true;
    n->live_obstacle_inflation_radius_=0.;
    n->hard_clearance_floor_=0.;
    n->terrain_speed_enable_=false;
    n->reverse_enable_=n->reverse_recovery_enable_=true;
    n->goal_reached_distance_=.12;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;
    n->odom_.stamp=n->now();
    n->odom_.v=-.08;
    n->have_goal_=n->have_path_=true;
    n->goal_seq_=1;
    n->raw_goal_={-.30,0.,0.,0.};
    n->path_={{0.,0.,0.,0.,-1},{-.15,0.,0.,.15,-1},{-.30,0.,0.,.30,-1}};

    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=100.;
    live->width=live->height=40;
    live->resolution=.2;
    live->min_x=live->min_y=-4.;
    live->hard.assign(1600,0);
    live->present_hard=live->hard;
    auto far=owned_actor(100.);
    far.center_x=2.5;
    far.center_y=2.5;
    far.cells={{{2.5,2.5}}};
    far.vx=.1;
    far.vy=0.;
    far.gaussian.mean<<2.5,2.5,.1,0.;
    live->tracks={far};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    terrain_nav::YieldState state;
    state.speed=n->odom_.v;
    const auto model=n->configuredYieldModel();
    const auto terminal=n->nominalInteractionCommands(n->odom_,state,model,4.5,*live,true);
    ASSERT_FALSE(terminal.empty());
    EXPECT_LT(terminal.front().speed,0.);
    const auto rollout=n->issuedYieldRollout(state,terminal,model,4.5,100.);
    ASSERT_FALSE(rollout.empty());
    EXPECT_NEAR(rollout.back().x,-.30,n->goal_reached_distance_);
    EXPECT_NEAR(rollout.back().speed,0.,1e-9);
    EXPECT_TRUE(n->nominalDynamicHandoverSafe(n->odom_,state,model,100.,4.5,*map,*live));
    n->recovery_path_=n->recovery_staged_gears_=true;
    EXPECT_TRUE(n->nominalDynamicHandoverSafe(n->odom_,state,model,100.,4.5,*map,*live));
    n->recovery_path_=n->recovery_staged_gears_=false;

    auto lateral=*live;
    lateral.tracks[0].center_x=-.2;
    lateral.tracks[0].center_y=.76;
    lateral.tracks[0].cells={{{-.2,.76}}};
    lateral.tracks[0].gaussian.mean<<-.2,.76,0.,0.;
    EXPECT_TRUE(n->earlierStoppingUnsafe(rollout,100.,lateral));
    EXPECT_TRUE(n->nominalDynamicHandoverSafe(n->odom_,state,model,100.,4.5,*map,lateral));

    auto blocked=*live;
    auto crossing=owned_actor(100.);
    crossing.center_x=-.6;
    crossing.center_y=1.25;
    crossing.cells={{{-.6,1.25}}};
    crossing.vy=-1.;
    crossing.gaussian.mean<<-.6,1.25,0.,-1.;
    blocked.tracks={crossing};
    EXPECT_FALSE(n->nominalDynamicHandoverSafe(n->odom_,state,model,100.,4.5,*map,blocked));
    auto unknown=*map;
    for(auto &slice:unknown.grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->nominalDynamicHandoverSafe(n->odom_,state,model,100.,4.5,unknown,*live));

    n->dynamic_active_=true;
    n->dynamic_goal_seq_=n->dynamic_conflict_goal_=1;
    n->dynamic_epoch_=n->input_epoch_.load();
    n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
    n->dynamic_origin_=n->dynamic_last_time_=100.;
    n->dynamic_horizon_sec_=4.5;
    n->dynamic_unresolved_[far.id]=100.;
    n->controlTimer();
    EXPECT_FALSE(n->dynamic_active_);
    EXPECT_TRUE(n->last_chance_accepted_)<<n->last_chance_reason_;
    EXPECT_LT(n->last_cmd_v_,0.);
  }

  static void nearby_goal_can_be_finished_during_a_dynamic_encounter_without_a_terminal_path()
  {
    auto n=node(1e-8F);
    n->motion_prediction_model_="gaussian_joint_cpu";
    n->motion_joint_risk_limit_=.05;
    n->live_obstacle_enable_=n->dynamic_acceleration_escape_enable_=true;
    n->reverse_enable_=n->reverse_recovery_enable_=true;
    n->live_obstacle_inflation_radius_=0.;
    n->hard_clearance_floor_=0.;
    n->goal_reached_distance_=.12;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;
    n->odom_.stamp=n->now();
    n->have_goal_=true;
    n->have_path_=false;
    n->goal_seq_=1;
    n->raw_goal_={.72,0.,0.,0.};
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=100.;
    live->width=live->height=40;
    live->resolution=.2;
    live->min_x=live->min_y=-4.;
    live->hard.assign(1600,0);
    live->present_hard=live->hard;
    auto crossing=owned_actor(100.);
    crossing.center_x=-.2;
    crossing.center_y=2.2;
    crossing.cells={{{-.2,2.2}}};
    crossing.vy=-.9;
    crossing.gaussian.mean<<-.2,2.2,0.,-.9;
    live->tracks={crossing};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
    EXPECT_GT(n->last_cmd_v_,0.);
    EXPECT_LE(n->last_cmd_v_,.6);
    const auto model=n->configuredYieldModel();
    const auto path=n->issuedYieldRollout(n->dynamic_initial_,n->dynamic_commands_,model,
      n->dynamic_horizon_sec_,n->dynamic_origin_);
    ASSERT_FALSE(path.empty());
    EXPECT_LT(std::hypot(path.back().x-.72,path.back().y),.12);
    EXPECT_NEAR(path.back().speed,0.,1e-9);
    for(auto &slice:map->grid["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->timedYieldSafe(path,100.,n->dynamic_commands_,*map,*live));
  }

  static void active_retreat_can_resume_after_an_interruption_only_with_fresh_proof()
  {
    auto n=node(1e-8F);
    n->motion_prediction_model_="gaussian_joint_cpu";
    n->motion_joint_risk_limit_=.05;
    n->live_obstacle_enable_=n->dynamic_acceleration_escape_enable_=true;
    n->reverse_enable_=n->reverse_recovery_enable_=true;
    n->live_obstacle_inflation_radius_=0.;
    n->hard_clearance_floor_=0.;
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
    n->odom_.valid=true;
    n->odom_.stamp=n->now();
    n->odom_.v=-.4;
    n->have_goal_=true;
    n->goal_seq_=n->dynamic_conflict_goal_=n->dynamic_goal_seq_=1;
    n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
    n->dynamic_epoch_=n->input_epoch_.load();
    n->raw_goal_={8.,0.,0.,0.};
    n->dynamic_active_=true;
    n->dynamic_last_optimization_time_=n->dynamic_origin_=n->dynamic_last_time_=99.5;
    n->dynamic_last_optimization_source_=99.5;
    n->dynamic_horizon_sec_=3.5;
    n->dynamic_commands_={{0.,-.5,0.},{1.5,0.,0.}};

    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=n->now();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=100.;
    live->width=live->height=40;
    live->resolution=.2;
    live->min_x=live->min_y=-4.;
    live->hard.assign(1600,0);
    live->present_hard=live->hard;
    auto far=owned_actor(100.);
    far.center_x=3.;
    far.center_y=2.;
    far.cells={{{3.,2.}}};
    far.gaussian.mean<<3.,2.,.1,0.;
    live->tracks={far};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
    EXPECT_EQ(n->dynamic_reoptimizations_,0u);
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,-.5);
    ASSERT_TRUE(n->last_retreat_.has_value());

    n->dynamic_active_=false;
    n->dynamic_commands_.clear();
    n->publishImmediateStop();
    ASSERT_EQ(rcl_set_ros_time_override(clock,100100000000LL),RCL_RET_OK);
    n->odom_.stamp=n->now();
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,-.5);
    EXPECT_NEAR(n->dynamic_origin_+n->dynamic_commands_.back().t,101.,1e-9);

    auto unknown=std::make_shared<ControlMapSnapshot>(*map);
    for(auto &slice:unknown->grid["probability_valid"])slice.setZero();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(unknown));
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_FALSE(n->last_chance_accepted_);
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
  }

  static void geometric_conflict_retirement_never_grants_unknown_ground_or_uses_missing_tracks()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=.2;
    const double time=n->now().seconds();
    LiveObstacleSnapshot live;live.stamp_sec=time;live.min_x=live.min_y=-4.;
    live.width=live.height=40;live.resolution=.2;live.hard.assign(1600,0);live.present_hard=live.hard;
    terrain_nav::MotionTrackSnapshot far;far.id=7;far.stamp=time;far.resolution=.05;
    far.cells={{{-3.,1.}}};far.vx=-.3;far.vy=-.6;
    far.circular=true;far.center_x=-3.;far.center_y=1.;far.radius_bound=.5;far.velocity_error=.1;
    terrain_nav::MotionTrackSnapshot crossing;crossing.id=8;crossing.stamp=time;crossing.resolution=.1;
    crossing.cells={{{0.,3.}}};crossing.vy=-1.5;
    n->dynamic_unresolved_[7]=time;n->dynamic_active_=true;live.tracks={far,crossing};
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    EXPECT_TRUE(n->dynamic_unresolved_.count(7)); // Another target still threatens the stop.
    live.tracks.clear();
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    far.stamp=time-1.;live.tracks={far};
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    EXPECT_TRUE(n->dynamic_unresolved_.count(7)); // Expiry/absence never releases it.
    far.stamp=time;live.tracks={far};
    auto unknown=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    unknown->stamp=n->now()-rclcpp::Duration::from_seconds(2.);
    for(auto &slice:unknown->grid["probability_valid"])slice.setZero();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(unknown));
    EXPECT_TRUE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    EXPECT_TRUE(n->dynamic_unresolved_.empty());EXPECT_FALSE(n->dynamic_active_);
    EXPECT_EQ(n->dynamic_geometric_retirements_,1u);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    // The dynamic bookkeeping can be resolved even though terrain is stale.
    // Neither stale nor fresh-UNKNOWN terrain can authorize a new command.
    geometry_msgs::msg::Twist request;request.linear.x=.5;
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(n->odom_,request));
    unknown->stamp=n->now();
    EXPECT_FALSE(n->commandBrakingEnvelopeSafe(n->odom_,request));
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    // A previous acceleration request can still act during the new stop's
    // delay. Measured zero speed alone would incorrectly clear this target.
    n->odom_.v=0.;far.circular=false;far.velocity_error=0.;far.vx=0.;far.vy=-.8;
    const auto envelope=n->collisionEnvelope(n->footprint_margin_);
    far.cells={{{envelope.front+.15,2.}}};live.tracks={far};
    n->dynamic_unresolved_[7]=time;n->last_cmd_v_=0.;
    EXPECT_TRUE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    n->dynamic_unresolved_[7]=time;n->last_cmd_v_=1.33;
    EXPECT_FALSE(n->retireGeometricallySeparatedConflicts(n->odom_,time,live));
    EXPECT_TRUE(n->dynamic_unresolved_.count(7));
  }

  static void dynamic_acceleration_reach_is_directional_and_time_indexed()
  {
    auto n=node(1e-8F);n->live_obstacle_inflation_radius_=0.;
    LiveObstacleSnapshot live;terrain_nav::MotionTrackSnapshot track;
    track.id=1;track.stamp=10.;track.resolution=.05;track.cells={{{0.,3.}}};track.vy=-.4;
    live.tracks={track};
    std::vector<terrain_nav::YieldState> stopped(2);stopped.back().t=2.;
    EXPECT_FALSE(n->timedMovingUnsafe(stopped,10.,live)); // CV remains far above the lane.
    live.tracks[0].acceleration_bound=1.2;
    EXPECT_TRUE(n->timedMovingUnsafe(stopped,10.,live)); // Physically accelerated crossing can reach it.
    for(auto &state:stopped)state.x=2.;
    EXPECT_FALSE(n->timedMovingUnsafe(stopped,10.,live)); // Not an isotropic alarm on nearby motion.
    live.tracks[0].acceleration_bound=NAN;
    EXPECT_TRUE(n->timedMovingUnsafe(stopped,10.,live));
  }

  static void dynamic_commit_rejects_replaced_expired_or_unsupported_certificates()
  {
    for(const std::string fault:{"none","forecast_only","current_occupied","map_replaced","live_replaced","stale_map","stale_live",
          "stale_odom","unsupported","nan_path","nan_brake","goal","epoch","localization"}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;n->have_goal_=true;n->goal_seq_=1;
      n->max_map_age_sec_=.5;
      auto *clock=n->get_clock()->get_clock_handle();
      ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
      auto live=std::make_shared<LiveObstacleSnapshot>();live->width=live->height=40;
      live->min_x=live->min_y=-4;live->resolution=.2;live->stamp_sec=100.;live->inflation_radius=0.;
      live->hard.assign(1600,0);live->present_hard=live->hard;
      if(fault=="forecast_only")live->hard.assign(1600,1);
      if(fault=="current_occupied")live->hard=live->present_hard=std::vector<uint8_t>(1600,1);
      OdomState odom;odom.valid=true;odom.stamp=n->now();
      if(fault=="stale_map")map->stamp=map->stamp-rclcpp::Duration::from_seconds(1.);
      if(fault=="stale_live")live->stamp_sec-=1.;
      if(fault=="stale_odom")odom.stamp=odom.stamp-rclcpp::Duration::from_seconds(1.);
      if(fault=="unsupported")for(auto &slice:map->grid["probability_valid"])slice.setZero();
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      if(fault=="map_replaced")std::atomic_store(&n->control_map_snapshot_,
        std::shared_ptr<const ControlMapSnapshot>(std::make_shared<ControlMapSnapshot>(*map)));
      if(fault=="live_replaced")std::atomic_store(&n->live_snapshot_,
        std::shared_ptr<const LiveObstacleSnapshot>(std::make_shared<LiveObstacleSnapshot>(*live)));
      n->dynamic_path_upper_=fault=="nan_path"?NAN:.001;
      n->dynamic_braking_upper_=fault=="nan_brake"?NAN:.001;
      if(fault=="goal")++n->goal_seq_;
      if(fault=="epoch")n->input_epoch_.fetch_add(1);
      if(fault=="localization")n->localization_fault_=true;
      geometry_msgs::msg::Twist cmd;cmd.linear.x=.3;
      const bool accepted=n->commitDynamicCommand(odom,cmd,1,0,map,live);
      const bool expected=fault=="none" || fault=="forecast_only";
      EXPECT_EQ(accepted,expected)<<fault;
      EXPECT_EQ(n->last_chance_accepted_,expected)<<fault;
      EXPECT_DOUBLE_EQ(n->last_cmd_v_,expected?.3:0.)<<fault;
      EXPECT_EQ(n->dynamic_yield_state_,expected?"advance":"refused")<<fault;
    }
  }

  static void normal_control_anticipates_compact_cue_before_velocity_confirmation()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->unclassified_encounter_speed_=.55;
    n->terrain_speed_enable_=false;
    n->odom_.valid=true;n->odom_.v=.6;n->odom_.stamp=n->now();
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3,0,0,0};
    n->have_path_=true;n->path_={{0,0,0,0},{1,0,0,1},{2,0,0,2},{3,0,0,3}};
    n->last_cmd_v_=1.2;n->last_cmd_time_sec_=n->now().seconds()-.02;
    auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=n->now().seconds();
    live->width=live->height=40;live->resolution=.2;live->min_x=live->min_y=-4.;
    live->hard.assign(1600,0);live->present_hard=live->hard;live->inflation_radius=0.;
    live->compact_observed={{{3.,2.,3.6,2.8}}};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto before=live->hard;const auto map=std::atomic_load(&n->control_map_snapshot_);
    n->controlTimer();
    EXPECT_GT(n->last_cmd_v_,0.);EXPECT_LE(n->last_cmd_v_,.55);
    EXPECT_TRUE(live->tracks.empty());EXPECT_EQ(live->hard,before);
    EXPECT_EQ(std::atomic_load(&n->control_map_snapshot_),map);
    // Caution may never override a real obstacle/unknown-ground veto.
    live->hard.assign(1600,1);live->present_hard=live->hard;
    n->controlTimer();EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
  }

  static void dynamic_yield_control_gates_and_cancellation()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->odom_.valid=true;n->odom_.stamp=n->now();n->have_goal_=true;n->goal_seq_=1;
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=n->now().seconds()-2;live->width=40;live->height=40;
    live->hard.assign(1600,0);live->present_hard=live->hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->dynamic_active_=true;n->dynamic_goal_seq_=1;
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_FALSE(n->dynamic_active_);EXPECT_EQ(n->last_cmd_v_,0);
    live->stamp_sec=n->now().seconds();n->dynamic_active_=true;n->dynamic_goal_seq_=0;
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_FALSE(n->dynamic_active_);
    n->dynamic_active_=true;n->dynamic_goal_seq_=1;n->dynamic_last_time_=n->now().seconds()+1;
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_FALSE(n->dynamic_active_);
    n->reverse_enable_=false;
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));EXPECT_EQ(n->last_cmd_v_,0);
  }

  static void completed_dynamic_maneuver_recertifies_the_handover_stop()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.v=-.02;
    n->have_goal_=true;n->goal_seq_=1;
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=n->now().seconds();live->min_x=live->min_y=-4;
    live->width=live->height=40;live->resolution=.2;
    live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot track;track.cells={{{3.,3.}}};
    track.vx=.5;track.stamp=live->stamp_sec;track.resolution=.2;live->tracks={track};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    const auto finish=[&] {
      n->dynamic_active_=true;n->dynamic_goal_seq_=1;n->dynamic_epoch_=n->input_epoch_.load();
      n->dynamic_origin_=n->now().seconds()-3.1;n->dynamic_last_time_=n->now().seconds()-.02;
      n->dynamic_braking_upper_=std::numeric_limits<double>::quiet_NaN();
      EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_FALSE(n->dynamic_active_);
      EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    };
    finish();
    EXPECT_TRUE(n->last_chance_accepted_);EXPECT_EQ(n->last_chance_reason_,"handover stop certified");
    EXPECT_TRUE(std::isfinite(n->dynamic_braking_upper_));
    EXPECT_LE(n->dynamic_braking_upper_,n->chance_braking_budget_);
    auto unsupported=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    for(auto &slice:unsupported->grid["probability_valid"])slice.setZero();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(unsupported));
    finish();EXPECT_FALSE(n->last_chance_accepted_);
  }

  static void unavailable_motion_does_not_invent_or_erase_geometric_conflicts()
  {
    for(int fault=0;fault<9;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;
      n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->have_goal_=true;n->goal_seq_=1;
      n->odom_.valid=true;n->odom_.stamp=n->now();const double time=n->now().seconds();
      auto live=std::make_shared<LiveObstacleSnapshot>();
      live->min_x=live->min_y=-4;live->width=live->height=40;live->resolution=.2;
      live->stamp_sec=time;live->hard.assign(1600,0);live->present_hard=live->hard;
      terrain_nav::MotionTrackSnapshot track;track.id=9;track.cells={{{15.,3.6}}};
      track.vx=1.5;track.resolution=.05;track.stamp=time;
      if(fault==0)track.stamp-=.61; // Actual failed-run remote-wall coast.
      if(fault==1)track.stamp+=.1;
      if(fault==2)track.vx=NAN;
      if(fault==3)track.cells.clear();
      if(fault==4)track.cells[0][0]=NAN;
      if(fault>=5) {
        n->motion_prediction_model_="gaussian_joint_cpu";
        track.circular=track.center_motion_fit=track.motion_confirmed=true;
        track.center_x=15.;track.center_y=3.6;track.radius_bound=.4;
        track.gaussian.valid=true;track.gaussian.stamp=time;
        track.gaussian.mean<<15.,3.6,1.5,0.;track.gaussian.covariance=Eigen::Matrix4d::Identity()*1e-6;
        if(fault==5)track.gaussian.stamp-=.1;
        if(fault==6)track.radius_bound=.001;
        if(fault==7)track.gaussian.stamp=track.stamp=time+.01;
        if(fault==8) {track.gaussian.valid=false;track.vx=0.;track.acceleration_bound=1.2;}
      }
      live->tracks={track};
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      n->last_cmd_v_=.5;
      EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      EXPECT_EQ(n->dynamic_yield_state_,"motion_input_hold");
      EXPECT_TRUE(n->dynamic_unresolved_.empty());EXPECT_FALSE(n->last_chance_accepted_);
      live->tracks.clear();
      EXPECT_FALSE(n->dynamicYieldControl(n->odom_));
      EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.); // New snapshot is not itself a command.
      // A previously confirmed conflict is NEVER cleared by that new input.
      n->dynamic_unresolved_[7]=time;
      EXPECT_TRUE(n->dynamicYieldControl(n->odom_));
      EXPECT_EQ(n->dynamic_yield_state_,"prediction_lost_hold");
      EXPECT_TRUE(n->dynamic_unresolved_.count(7));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
    }
  }

  static void observation_limited_motion_cannot_latch_missing_ids_or_clear_obstacles()
  {
    auto n=node(1e-8F);n->dynamic_observed_only_=true;n->live_obstacle_enable_=true;
    n->live_obstacle_inflation_radius_=0.;n->reverse_recovery_enable_=n->reverse_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
    n->odom_.valid=true;n->odom_.stamp=n->now();const double time=n->now().seconds();
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=live->min_y=-4;live->width=live->height=40;live->resolution=.2;
    live->stamp_sec=time;live->hard.assign(1600,0);live->present_hard=live->hard;
    live->hard[20*40+20]=live->present_hard[20*40+20]=1;
    const auto occupied=live->present_hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->dynamic_conflict_goal_=1;n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
    n->dynamic_unresolved_[99]=time-.3;
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));
    EXPECT_TRUE(n->dynamic_unresolved_.empty());EXPECT_EQ(n->dynamic_unobserved_conflicts_skipped_,1u);
    EXPECT_EQ(live->present_hard,occupied);EXPECT_EQ(live->hard,occupied);
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);EXPECT_FALSE(n->last_chance_accepted_);
    terrain_nav::YieldState root;terrain_nav::YieldModel model;
    const auto stop=terrain_nav::rolloutYield(root,root,{{0,0,0}},model,3.);
    EXPECT_FALSE(n->timedYieldSafe(stop,time,{{0,0,0}},*std::atomic_load(&n->control_map_snapshot_),*live));
    // A sensor/map outage is not an observation-limited source update.
    n->dynamic_unresolved_[98]=time-.3;live->stamp_sec=time-2.;
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));EXPECT_EQ(n->dynamic_unresolved_.count(98),1u);
    live->stamp_sec=time;
    // Losing current observations during an already selected maneuver hands
    // back to ordinary control. It neither publishes the old maneuver nor
    // clears the real occupied cell under the vehicle.
    n->dynamic_active_=true;n->dynamic_commands_={{0.,.4,.1},{2.,0.,0.}};
    n->dynamic_origin_=time;n->dynamic_goal_seq_=n->goal_seq_;
    n->dynamic_epoch_=n->input_epoch_.load();n->dynamic_last_time_=time;
    n->dynamic_horizon_sec_=3.;
    const auto queue_before=n->issuedCommandSnapshot();
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));EXPECT_FALSE(n->dynamic_active_);
    EXPECT_EQ(n->dynamic_yield_state_,"idle");
    EXPECT_EQ(n->issuedCommandSnapshot().size(),queue_before.size());
    EXPECT_EQ(live->present_hard,occupied);EXPECT_EQ(live->hard,occupied);
    EXPECT_FALSE(n->timedYieldSafe(stop,time,{{0,0,0}},*std::atomic_load(&n->control_map_snapshot_),*live));
    n->dynamic_unresolved_[98]=time-.3;
    terrain_nav::MotionTrackSnapshot seen;seen.id=7;seen.stamp=time;seen.resolution=.1;
    seen.cells={{{0.,0.}}};seen.vy=-.8;live->tracks={seen};n->dynamic_unresolved_[7]=time;
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_EQ(n->dynamic_unresolved_.count(7),1u);EXPECT_EQ(n->dynamic_unresolved_.count(98),0u);
    EXPECT_NE(n->dynamic_yield_state_,"prediction_lost_hold");
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);EXPECT_FALSE(n->last_chance_accepted_);
    EXPECT_EQ(live->present_hard,occupied);
  }

  static void lost_dynamic_prediction_does_not_resume_on_the_next_tick()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;n->have_goal_=true;n->goal_seq_=1;
    n->raw_goal_={3.,0.,0.,0.};n->odom_.valid=true;n->odom_.stamp=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();const double time=n->now().seconds();
    live->min_x=live->min_y=-4;live->width=live->height=40;live->resolution=.2;
    live->stamp_sec=time;live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot track;track.id=7;track.cells={{{-.65,3.}}};
    track.vy=-1.5;track.stamp=time;track.resolution=.1;live->tracks={track};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));ASSERT_TRUE(n->dynamic_unresolved_.count(7));
    live->tracks.clear();
    for(int tick=0;tick<4;++tick) {
      ASSERT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      EXPECT_EQ(n->dynamic_yield_state_,"prediction_lost_hold");
      EXPECT_TRUE(n->dynamic_unresolved_.count(7));
    }
    // A different stationary object cannot resolve the lost target.
    auto tentative=track;tentative.id=0;tentative.motion_confirmed=false;tentative.center_motion_fit=true;
    live->tracks={tentative};
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_EQ(n->dynamic_yield_state_,"prediction_lost_hold");
    EXPECT_TRUE(n->dynamic_unresolved_.count(7));live->tracks.clear();
    track.id=8;track.vy=0;track.stamp=time+.001;live->resolved_tracks={track};
    EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_TRUE(n->dynamic_unresolved_.count(7));
    track.id=7;live->resolved_tracks={track};
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));EXPECT_TRUE(n->dynamic_unresolved_.empty());
    EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.); // Release is not itself a motion command.
  }

  static void stopped_interaction_hands_back_without_renewing_its_stationary_tail_forever()
  {
    for(int fault=0;fault<3;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
      n->reverse_recovery_enable_=n->reverse_enable_=true;n->dynamic_acceleration_escape_enable_=true;
      n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;n->hard_clearance_floor_=0.;
      auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
      if(fault==1)for(auto &slice:map->grid["probability_valid"])slice.setZero();
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=100.;live->resolution=.2;
      live->width=live->height=40;live->min_x=live->min_y=-4.;
      live->present_hard.assign(1600,0);live->hard=live->present_hard;
      auto actor=owned_actor(100.);actor.center_x=actor.center_y=-3.;actor.cells={{{-3.,-3.}}};
      actor.gaussian.mean<<-3.,-3.,-.1,0.;actor.vx=-.1;actor.vy=0.;actor.radius_bound=.12;
      actor.gaussian.covariance.setZero();actor.gaussian.process_density=0.;live->tracks={actor};
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      n->odom_.valid=true;n->odom_.stamp=n->now();n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
      n->dynamic_active_=true;n->dynamic_goal_seq_=1;n->dynamic_epoch_=n->input_epoch_.load();
      n->dynamic_origin_=n->dynamic_last_time_=99.4;n->dynamic_horizon_sec_=3.;
      n->dynamic_last_optimization_time_=99.4;n->dynamic_last_optimization_source_=99.4;
      n->dynamic_commands_={{0.,.6,.1},{.2,0.,0.}};
      if(fault==2){n->dynamic_commands_.push_back({.8,.4,0.});n->dynamic_commands_.push_back({1.4,0.,0.});}
      n->issued_commands_.record({99.6,0.,0.});
      const bool consumed=n->dynamicYieldControl(n->odom_);
      if(fault==0) {
        EXPECT_FALSE(consumed);EXPECT_FALSE(n->dynamic_active_);EXPECT_EQ(n->dynamic_yield_state_,"idle");
        EXPECT_EQ(n->dynamic_reoptimizations_,0u);
      } else if(fault==1) {
        EXPECT_TRUE(consumed);EXPECT_FALSE(n->last_chance_accepted_);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      } else {EXPECT_TRUE(consumed);EXPECT_TRUE(n->dynamic_active_);EXPECT_TRUE(n->last_chance_accepted_);}
    }
  }

  static void receding_interaction_can_improve_a_still_safe_arc_without_reusing_its_old_permission()
  {
    for(int fault=0;fault<5;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
      n->dynamic_acceleration_escape_enable_=true;n->hard_clearance_floor_=0.;
      n->reverse_recovery_enable_=n->reverse_enable_=true;
      n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
      n->wheel_base_=.43;n->rear_axle_offset_=.215;n->max_steer_=.31432913231137;
      n->max_speed_=1.33;n->dynamic_yield_delay_=.04;n->dynamic_anticipation_horizon_sec_=4.5;
      auto *clock=n->get_clock()->get_clock_handle();ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));map->stamp=n->now();
      if(fault==1)for(auto &slice:map->grid["probability_valid"])slice.setZero();
      if(fault==4) {
        const auto size=map->grid.getSizePos();
        for(int r=0;r<size[0];++r)for(int c=0;c<size[1];++c) {
          Eigen::Vector3d p;map->grid.index2Pos(Eigen::Array3i(r,c,0),p);
          if(p.x()<-.05||p.x()>.5||std::abs(p.y())>.6)
            for(auto &slice:map->grid["probability_valid"])slice(r,c)=0.;
        }
      }
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      auto live=std::make_shared<LiveObstacleSnapshot>();live->stamp_sec=100.;
      live->width=live->height=40;live->resolution=.2;live->min_x=live->min_y=-4.;
      live->present_hard.assign(1600,0);live->hard=live->present_hard;
      auto actor=owned_actor(100.);actor.center_x=actor.center_y=-3.;actor.radius_bound=.12;
      actor.vx=-.1;actor.vy=0.;actor.cells={{{-3.,-3.}}};actor.gaussian.mean<<-3.,-3.,-.1,0.;
      actor.gaussian.covariance.setZero();actor.gaussian.process_density=0.;live->tracks={actor};
      if(fault==2){n->dynamic_unresolved_={{actor.id,99.5}};live->tracks.clear();}
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      n->odom_.valid=true;n->odom_.stamp=n->now();n->odom_.yaw=.4;n->odom_.v=.6;
      n->odom_.yaw_rate=.6*std::tan(n->max_steer_)/n->wheel_base_;
      n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
      n->dynamic_active_=true;n->dynamic_goal_seq_=1;n->dynamic_epoch_=n->input_epoch_.load();
      n->dynamic_conflict_goal_=1;n->dynamic_conflict_spatial_=n->spatial_epoch_.load();
      n->dynamic_origin_=n->dynamic_last_time_=99.5;n->dynamic_horizon_sec_=3.;
      n->dynamic_last_optimization_time_=99.5;n->dynamic_last_optimization_source_=fault==3?100.:99.5;
      n->dynamic_commands_={{0.,.6,n->max_steer_},{fault==4?.8:1.7,0.,n->max_steer_}};
      const auto original=n->dynamic_commands_;
      n->issued_commands_.record({99.,.6,n->max_steer_});
      terrain_nav::YieldModel model;model.wheelbase=.43;model.rear_axle_offset=.215;model.delay=.04;
      const terrain_nav::YieldState root{0.,0.,0.,.4,.6,n->max_steer_};
      const auto warm=terrain_nav::remainingYieldCommands(original,.5);
      const auto previous=terrain_nav::finiteJointRollout(n->issuedYieldRollout(root,warm,model,6.,100.));
      const double old_preference=n->interactionPolicyUtility(root,previous,n->raw_goal_);
      if(fault==4)n->dynamic_before_selection_proof_test_hook_=[&] {
        ASSERT_EQ(rcl_set_ros_time_override(clock,100050000000LL),RCL_RET_OK);
        auto replacement=std::make_shared<ControlMapSnapshot>(*map);replacement->stamp=n->now();
        std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(replacement));
        auto new_live=std::make_shared<LiveObstacleSnapshot>(*live);new_live->stamp_sec=100.05;
        new_live->tracks[0].stamp=new_live->tracks[0].gaussian.stamp=100.05;
        std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(new_live));
        n->odom_.stamp=n->now();n->odom_.x=.03*std::cos(.4);n->odom_.y=.03*std::sin(.4);
        n->odom_.yaw=.4+.05*n->odom_.yaw_rate;
      };
      ASSERT_TRUE(n->dynamicYieldControl(n->odom_))<<fault;
      if(fault==0) {
        EXPECT_EQ(n->dynamic_reoptimizations_,1u)<<n->dynamic_yield_reason_;
        ASSERT_TRUE(n->dynamic_active_)<<n->dynamic_yield_reason_;EXPECT_TRUE(n->last_chance_accepted_);
        const auto selected=terrain_nav::finiteJointRollout(n->issuedYieldRollout(root,n->dynamic_commands_,model,6.,100.));
        EXPECT_GT(n->interactionPolicyUtility(root,selected,n->raw_goal_),old_preference+.01);
        EXPECT_LT(std::abs(selected.back().yaw),std::abs(previous.back().yaw));
        EXPECT_TRUE(n->timedYieldSafe(selected,100.,n->dynamic_commands_,*map,*live));
      } else if(fault==4) {
        ASSERT_TRUE(n->last_chance_accepted_)<<n->dynamic_yield_reason_;
        EXPECT_EQ(n->dynamic_reoptimizations_,1u);EXPECT_EQ(n->dynamic_selection_input_refreshes_,1u);
        ASSERT_EQ(n->dynamic_commands_.size(),2u);
        EXPECT_NEAR(n->dynamic_origin_+n->dynamic_commands_.back().t,100.3,1e-10);
        EXPECT_DOUBLE_EQ(n->dynamic_commands_.front().speed,.6);
      } else if(fault==3) {
        EXPECT_EQ(n->dynamic_reoptimizations_,0u);EXPECT_TRUE(n->last_chance_accepted_);
        EXPECT_DOUBLE_EQ(n->dynamic_origin_,99.5);EXPECT_EQ(n->dynamic_commands_.size(),original.size());
      } else {
        EXPECT_FALSE(n->last_chance_accepted_);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      }
    }
  }

  static void interaction_progress_accounts_for_the_actual_turning_radius_and_terminal_heading()
  {
    auto n=node(1e-8F);n->wheel_base_=.43;n->rear_axle_offset_=.215;n->max_steer_=.31432913231137;
    terrain_nav::YieldState root;Pose2 goal{10.,0.,0.,0.};
    EXPECT_NEAR(n->interactionRemainingDistance(root,goal),10.,1e-10);
    root.x=3.;EXPECT_NEAR(n->interactionRemainingDistance(root,goal),7.,1e-10);
    auto turned=root;turned.yaw=M_PI/2;
    const double turning=n->interactionRemainingDistance(turned,goal);
    EXPECT_GT(turning,7.5); // Euclidean distance alone calls these equivalent.
    const double radius=n->wheel_base_/std::tan(n->max_steer_);
    const auto rear_start=terrain_nav::rearAxlePose(Pose2{turned.x,turned.y,turned.yaw,0.},n->rear_axle_offset_);
    const auto rear_goal=terrain_nav::rearAxlePose(goal,n->rear_axle_offset_);
    EXPECT_GE(turning,std::hypot(rear_goal.x-rear_start.x,rear_goal.y-rear_start.y));
    n->max_steer_*=.7;EXPECT_GE(n->interactionRemainingDistance(turned,goal),turning-1e-10);
    n->max_steer_=.31432913231137;
    const double angle=.71,c=std::cos(angle),s=std::sin(angle);
    auto transformed=turned;transformed.x=4.+c*turned.x-s*turned.y;transformed.y=-2.+s*turned.x+c*turned.y;transformed.yaw+=angle;
    auto transformed_goal=goal;transformed_goal.x=4.+c*goal.x-s*goal.y;
    transformed_goal.y=-2.+s*goal.x+c*goal.y;transformed_goal.yaw+=angle;
    EXPECT_NEAR(n->interactionRemainingDistance(transformed,transformed_goal),turning,1e-10);
    // Ordering is not permission: it never queries or fills the terrain map.
    terrain_nav::YieldState initial;terrain_nav::YieldModel model;model.wheelbase=.43;model.rear_axle_offset=.215;model.delay=.04;
    const auto arc=terrain_nav::makeAdvanceCommands(initial,model,2.4,1.33,n->max_steer_,6.);
    const auto bypass=terrain_nav::makeBypassCommands(initial,model,2.4,1.33,n->max_steer_,.6,6.);
    ASSERT_FALSE(arc.empty());ASSERT_FALSE(bypass.empty());
    const auto a=terrain_nav::rolloutYield(initial,initial,arc,model,6.);
    const auto b=terrain_nav::rolloutYield(initial,initial,bypass,model,6.);
    ASSERT_GT(std::abs(a.back().yaw),1.);EXPECT_LT(std::abs(b.back().yaw),.1);
    EXPECT_GT(n->interactionPolicyUtility(initial,b,goal),n->interactionPolicyUtility(initial,a,goal));
    const double preference=n->interactionPolicyUtility(initial,b,goal);
    auto unknown=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    for(auto &slice:unknown->grid["probability_valid"])slice.setZero();
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(unknown));
    EXPECT_DOUBLE_EQ(n->interactionPolicyUtility(initial,b,goal),preference);
    LiveObstacleSnapshot live;live.stamp_sec=n->now().seconds();
    EXPECT_FALSE(n->timedYieldSafe(b,n->now().seconds(),bypass,*unknown,live));
    auto invalid=goal;invalid.yaw=NAN;EXPECT_EQ(n->interactionPolicyUtility(initial,b,invalid),-kInf);
    EXPECT_GT(radius,1.3);EXPECT_LT(radius,1.34); // Rear, not centre trajectory radius.
  }

  static void dynamic_avoidance_prefers_certified_forward_progress(bool invalid_old_stop=false,bool ordinary_pause=false,
                                                                  bool bounded_frontier=false,bool unconfirmed=false,
                                                                  bool cautious_approach=false,bool rear_goal=false)
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;
    n->dynamic_acceleration_escape_enable_=true; // Explicit experimental acceleration-mode regression.
    // Use the full Hunter body and its single margin, matching the physical
    // qualification. The generic node default has an additional 0.45m live
    // inflation intended for legacy setups, not this body's explicit envelope.
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.40;
    n->footprint_margin_=.10;n->live_obstacle_inflation_radius_=0.;n->hard_clearance_floor_=0.;
    n->reverse_recovery_enable_=true;n->reverse_enable_=true;
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
    if(rear_goal)n->raw_goal_={-3.,0.,M_PI,0.};
    auto live=std::make_shared<LiveObstacleSnapshot>();
    const double time=n->now().seconds();
    live->min_x=-4;live->min_y=-4;live->resolution=.2;live->width=40;live->height=40;
    live->stamp_sec=time;live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot track;
    track.cells={{{-.65,3.}}};track.vy=-1.5;track.stamp=time;track.resolution=.1;
    if(unconfirmed){track.motion_confirmed=false;track.center_motion_fit=true;n->center_candidate_actions_=true;}
    live->tracks={track};
    if(cautious_approach) {
      n->unclassified_encounter_speed_=.6;
      live->compact_observed={{{.5,1.,1.2,1.7}}};
    }
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    if(bounded_frontier) {
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      const auto size=map->grid.getSizePos();
      for(int r=0;r<size[0];++r)for(int c=0;c<size[1];++c) {
        Eigen::Vector3d p;map->grid.index2Pos(Eigen::Array3i(r,c,0),p);
        if(p.x()>1.65 || std::abs(p.y())>.6)
          for(auto &slice:map->grid["probability_valid"])slice(r,c)=0;
      }
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      terrain_nav::YieldState initial;terrain_nav::YieldModel model;
      for(double length:{1.8,3.}) {
        const auto commands=terrain_nav::makeAdvanceCommands(initial,model,length,.6,0.,6.);
        ASSERT_FALSE(commands.empty());
        const auto rollout=terrain_nav::rolloutYield(initial,initial,commands,model,6.);
        EXPECT_FALSE(n->timedYieldSafe(rollout,time,commands,*map,*live));
      }
    }
    if(invalid_old_stop) {
      n->dynamic_active_=true;n->dynamic_goal_seq_=n->goal_seq_;
      n->dynamic_epoch_=n->input_epoch_.load();n->dynamic_origin_=time;n->dynamic_last_time_=time;
      n->dynamic_horizon_sec_=4.;n->dynamic_initial_={};n->dynamic_commands_={{0,0,0}};
      const auto old=terrain_nav::rolloutYield(n->dynamic_initial_,n->dynamic_initial_,n->dynamic_commands_,{},4.);
      ASSERT_TRUE(n->timedMovingUnsafe(old,time,*live));
    }
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_,ordinary_pause));
    EXPECT_TRUE(n->dynamic_active_);EXPECT_GT(n->last_cmd_v_,0);
    EXPECT_EQ(n->dynamic_yield_state_,"advance");
    EXPECT_LE(n->dynamic_path_upper_,n->chance_path_budget_);
    EXPECT_LE(n->dynamic_braking_upper_,n->chance_braking_budget_);
    if(cautious_approach) {
      EXPECT_LE(n->last_cmd_v_,.6);
      // An old high-speed action cannot bypass a newly applicable limit.
      n->dynamic_commands_.front().speed=n->max_speed_;
      ASSERT_TRUE(n->dynamicYieldControl(n->odom_,ordinary_pause));
      EXPECT_LE(n->last_cmd_v_,.6);
      for(const auto &command:n->dynamic_commands_)EXPECT_LE(command.speed,.6);
    }
    if(unconfirmed) {
      EXPECT_TRUE(n->dynamic_unresolved_.empty()); // No permanent identity invented.
      n->center_candidate_actions_=false;
      EXPECT_TRUE(n->dynamicYieldControl(n->odom_));EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
      EXPECT_EQ(n->dynamic_yield_state_,"motion_input_hold");
      n->center_candidate_actions_=true;
    }
    if(bounded_frontier) {
      terrain_nav::YieldModel model;
      const auto chosen=terrain_nav::rolloutYield(n->dynamic_initial_,n->dynamic_initial_,n->dynamic_commands_,model,n->dynamic_horizon_sec_);
      ASSERT_FALSE(chosen.empty());
      EXPECT_GT(chosen.back().x,.4);EXPECT_LT(chosen.back().x,1.7);
      EXPECT_NEAR(chosen.back().speed,0.,.001);
    }
    // With both standing and rear travel threatened, the default cannot
    // advertise an unqualified acceleration escape as an accepted command.
    n->dynamic_active_=false;n->dynamic_acceleration_escape_enable_=false;
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_FALSE(n->dynamic_active_);EXPECT_EQ(n->last_cmd_v_,0.);
    EXPECT_EQ(n->dynamic_yield_state_,"refused");
    // A newly observed solid wall must cancel the selected advance, rather
    // than allowing the forecast maneuver to override current occupancy.
    std::fill(live->present_hard.begin(),live->present_hard.end(),1);
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_FALSE(n->dynamic_active_);EXPECT_EQ(n->last_cmd_v_,0.);
  }

  static void anticipation_can_start_a_certified_action_before_the_short_stop_becomes_unsafe()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->dynamic_acceleration_escape_enable_=true;n->reverse_recovery_enable_=n->reverse_enable_=true;
    n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
    n->hard_clearance_floor_=0.;n->max_speed_=1.33;
    n->odom_.valid=true;n->odom_.stamp=n->now();n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4;
    live->width=live->height=40;live->resolution=.2;live->stamp_sec=n->now().seconds();
    live->hard.assign(1600,0);live->present_hard=live->hard;
    terrain_nav::MotionTrackSnapshot actor;actor.id=1;actor.circular=actor.center_motion_fit=true;
    actor.cells={{{0,3.7}}};actor.center_y=3.7;actor.radius_bound=.20;actor.resolution=.05;
    actor.vy=-.9;actor.stamp=live->stamp_sec;actor.gaussian.valid=true;actor.gaussian.stamp=actor.stamp;
    actor.gaussian.mean<<0.,3.7,0.,-.9;actor.gaussian.covariance=Eigen::Matrix4d::Identity()*1e-12;
    actor.gaussian.process_density=0.;live->tracks={actor};
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->dynamic_anticipation_horizon_sec_=3.;
    EXPECT_FALSE(n->dynamicYieldControl(n->odom_));EXPECT_FALSE(n->dynamic_active_);
    // No motion/goal/threshold changes: only earlier evidence of the SAME
    // crossing, still requiring the full independent action/terrain proof.
    n->dynamic_anticipation_horizon_sec_=4.5;
    ASSERT_TRUE(n->dynamicYieldControl(n->odom_));
    EXPECT_TRUE(n->last_chance_accepted_);EXPECT_TRUE(n->dynamic_active_);
    EXPECT_GT(n->last_cmd_v_,0.);EXPECT_LE(n->dynamic_braking_upper_,n->chance_braking_budget_);
    // Broad clear terrain should not require checking every surviving
    // terminal before proving a usable action. A terminal check itself is
    // still mandatory; no screen result grants execution permission.
    EXPECT_GT(n->dynamic_terminal_candidates_checked_,0);
    EXPECT_LT(n->dynamic_terminal_candidates_checked_,n->dynamic_screen_survivors_);
    EXPECT_LE(n->dynamic_terminal_candidates_checked_,12);
  }

  static void replaced_commit_inputs_require_a_new_bounded_action_proof()
  {
    for(int fault=0;fault<3;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;
      n->dynamic_acceleration_escape_enable_=true;
      n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.40;
      n->footprint_margin_=.10;n->live_obstacle_inflation_radius_=0.;n->hard_clearance_floor_=0.;
      n->reverse_recovery_enable_=true;n->reverse_enable_=true;
      auto *clock=n->get_clock()->get_clock_handle();
      ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      n->odom_.valid=true;n->odom_.stamp=n->now();
      n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      map->stamp=n->now();
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      auto live=std::make_shared<LiveObstacleSnapshot>();
      live->min_x=live->min_y=-4;live->width=live->height=40;live->resolution=.2;
      live->stamp_sec=100.;live->hard.assign(1600,0);live->present_hard=live->hard;
      terrain_nav::MotionTrackSnapshot actor;actor.cells={{{-.65,3.}}};actor.vy=-1.5;
      actor.stamp=100.;actor.resolution=.1;live->tracks={actor};
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      unsigned replacements=0;std::shared_ptr<ControlMapSnapshot> replaced;
      n->dynamic_before_commit_test_hook_=[&] {
        ++replacements;
        // This is a real immutable-map replacement AFTER action selection,
        // not a hand-authored choice of an executable output command.
        replaced=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
        const auto size=replaced->grid.getSizePos();
        for(int r=0;r<size[0];++r)for(int c=0;c<size[1];++c) {
          Eigen::Vector3d p;replaced->grid.index2Pos(Eigen::Array3i(r,c,0),p);
          if(fault==1 || (fault==2 && replacements==2) || p.x()>1.65 || std::abs(p.y())>.6)
            for(auto &slice:replaced->grid["probability_valid"])slice(r,c)=0;
        }
        std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(replaced));
      };
      ASSERT_TRUE(n->dynamicYieldControl(n->odom_))<<fault;
      EXPECT_EQ(n->dynamic_input_reselections_,1u)<<fault;
      EXPECT_LE(replacements,2u)<<fault;
      if(fault==0) {
        EXPECT_TRUE(n->last_chance_accepted_);EXPECT_GT(n->last_cmd_v_,0.);
        EXPECT_EQ(replacements,2u);
        terrain_nav::YieldModel model;
        const auto chosen=terrain_nav::rolloutYield(n->dynamic_initial_,n->dynamic_initial_,n->dynamic_commands_,model,n->dynamic_horizon_sec_);
        ASSERT_FALSE(chosen.empty());EXPECT_GT(chosen.back().x,.4);EXPECT_LT(chosen.back().x,1.7);
        EXPECT_TRUE(n->timedYieldSafe(chosen,100.,n->dynamic_commands_,*replaced,*live));
      } else {
        EXPECT_FALSE(n->last_chance_accepted_);EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.);
        EXPECT_FALSE(n->dynamic_active_);
        if(fault==2)EXPECT_EQ(replacements,2u); // No unbounded retries on churn.
      }
    }
  }

  static void absolute_clock_roundoff_cannot_invalidate_an_admitted_publication_boundary()
  {
    std::vector<OdomState> predicted;
    for(int64_t stamp_ns:{65014998547LL,30014998547LL}) {
      auto n=node(1e-8F);n->dynamic_yield_delay_=.04;n->wheel_base_=.43;n->rear_axle_offset_=.215;
      n->max_accel_=n->dynamic_yield_deceleration_=1.2;n->max_steer_=.31432913231137;
      auto *clock=n->get_clock()->get_clock_handle();
      ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,stamp_ns+135000000LL),RCL_RET_OK);
      const double source=double(stamp_ns)*1e-9;
      const double boundary=double(stamp_ns-40000000LL)*1e-9;
      if(stamp_ns==65014998547LL)ASSERT_LT(boundary-source,-.04); // Recorded failure's arithmetic.
      n->issued_commands_.record({double(stamp_ns-80000000LL)*1e-9,0.,0.});
      n->issued_commands_.record({boundary,.4,.12});
      n->issued_commands_.record({double(stamp_ns+20000000LL)*1e-9,.8,.12});
      n->issued_commands_.record({double(stamp_ns+65000000LL)*1e-9,1.,-.1});
      OdomState odom;odom.valid=true;odom.stamp=rclcpp::Time(stamp_ns,RCL_ROS_TIME);odom.v=.1;
      n->predictOdomForControl(odom,false);
      ASSERT_TRUE(odom.valid);EXPECT_TRUE(odom.command_prediction);
      EXPECT_EQ(odom.stamp.nanoseconds(),stamp_ns);EXPECT_NEAR(odom.prediction_sec,.135,1e-10);
      EXPECT_NEAR(odom.v,.262,1e-9);EXPECT_GT(odom.x,.02);EXPECT_LT(odom.x,.03);
      predicted.push_back(odom);
    }
    ASSERT_EQ(predicted.size(),2u);
    EXPECT_NEAR(predicted[0].x,predicted[1].x,1e-10);
    EXPECT_NEAR(predicted[0].y,predicted[1].y,1e-10);
    EXPECT_NEAR(predicted[0].yaw,predicted[1].yaw,1e-10);
  }

  static void selection_refreshes_sources_and_actual_root_before_complete_proofs()
  {
    for(int fault=0;fault<7;++fault) {
      auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
      n->dynamic_acceleration_escape_enable_=true;n->reverse_recovery_enable_=n->reverse_enable_=true;
      n->motion_prediction_model_="gaussian_joint_cpu";n->motion_joint_risk_limit_=.05;
      n->hard_clearance_floor_=0.;n->max_speed_=1.33;n->max_map_age_sec_=.5;
      n->dynamic_anticipation_horizon_sec_=4.5;
      auto *clock=n->get_clock()->get_clock_handle();
      ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
      ASSERT_EQ(rcl_set_ros_time_override(clock,100000000000LL),RCL_RET_OK);
      n->odom_.valid=true;n->odom_.stamp=n->now();
      n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={3.,0.,0.,0.};
      auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      map->stamp=rclcpp::Time(99900000000LL,RCL_ROS_TIME);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
      auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4;
      live->width=live->height=40;live->resolution=.2;live->stamp_sec=100.;
      live->hard.assign(1600,0);live->present_hard=live->hard;
      terrain_nav::MotionTrackSnapshot actor;actor.id=7;actor.circular=actor.center_motion_fit=true;
      actor.cells={{{0,3.7}}};actor.center_y=3.7;actor.radius_bound=.20;actor.resolution=.05;
      actor.vy=-.9;actor.stamp=100.;actor.gaussian.valid=true;actor.gaussian.stamp=100.;
      actor.gaussian.mean<<0.,3.7,0.,-.9;actor.gaussian.covariance=Eigen::Matrix4d::Identity()*1e-12;
      actor.gaussian.process_density=0.;
      // Every OLD-source path overlaps this model at its initial pose.
      // A newer observation replaces it before screening. Old-source
      // rejection must not erase all proposals before that refresh happens.
      if(fault==6) {
        actor.center_y=.3;actor.cells={{{0,.3}}};actor.gaussian.mean[1]=.3;
      }
      live->tracks={actor};
      std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
      unsigned replacements=0;
      n->dynamic_before_selection_proof_test_hook_=[&] {
        ++replacements;ASSERT_EQ(rcl_set_ros_time_override(clock,100460000000LL),RCL_RET_OK);
        auto next=std::make_shared<ControlMapSnapshot>(*map);next->stamp=rclcpp::Time(100360000000LL,RCL_ROS_TIME);
        if(fault==1)for(auto &slice:next->grid["probability_valid"])slice.setZero();
        if(fault==4)next->stamp=rclcpp::Time(100900000000LL,RCL_ROS_TIME);
        std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(next));
        auto observed=std::make_shared<LiveObstacleSnapshot>(*live);observed->stamp_sec=100.36;
        auto &a=observed->tracks.front();a.stamp=a.gaussian.stamp=100.36;
        a.center_y=3.7-.9*.36;a.cells={{{0,a.center_y}}};a.gaussian.mean[1]=a.center_y;
        if(fault==2||fault==3)observed->tracks.clear();
        if(fault==3){actor.stamp=100.36;actor.vy=0.;observed->resolved_tracks={actor};}
        std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(observed));
        n->odom_.stamp=n->now();n->odom_.x=.07;n->odom_.v=0.;
      };
      if(fault==5)n->dynamic_before_commit_test_hook_=[&] {
        auto missing=std::make_shared<LiveObstacleSnapshot>(*std::atomic_load(&n->live_snapshot_));
        missing->tracks.clear();
        std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(missing));
      };
      const auto original_root=n->odom_;
      ASSERT_TRUE(n->dynamicYieldControl(original_root))<<fault;
      EXPECT_EQ(replacements,1u)<<fault;EXPECT_EQ(n->dynamic_selection_input_refreshes_,1u)<<fault;
      if(fault==0||fault==3||fault==6) {
        EXPECT_TRUE(n->last_chance_accepted_)<<fault;EXPECT_GT(n->last_cmd_v_,0.)<<fault;
        EXPECT_NEAR(n->dynamic_initial_.x,.07,1e-10);EXPECT_NEAR(n->dynamic_origin_,100.46,1e-9);
        EXPECT_GT(n->dynamic_origin_-map->stamp.seconds(),.5); // Old generation really expired.
        EXPECT_NEAR(n->motion_joint_last_origin_s_.load(),100.46,1e-9);
      } else {EXPECT_FALSE(n->last_chance_accepted_)<<fault;EXPECT_DOUBLE_EQ(n->last_cmd_v_,0.)<<fault;}
    }
  }

  static void unknown_frontier_does_not_invent_a_sideways_detour()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->chance_allow_startup_prior_=true;
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    n->map_.add("probability_pose_event",true,1);
    for(int r=0;r<40;++r) for(int c=0;c<40;++c) {
      Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
      if(p.x()>1.5 && std::abs(p.y())<1.)
        for(auto &slice:n->map_["probability_valid"])slice(r,c)=0;
    }
    n->rebuildCostmapLocked();
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_={3.,0.,0.,0.};
    ASSERT_TRUE(n->planPath("aligned unknown frontier"));
    ASSERT_FALSE(n->path_.empty());
    EXPECT_GE(n->path_.back().s,.25);
    EXPECT_TRUE(std::all_of(n->path_.begin(),n->path_.end(),[](const auto &p){return std::abs(p.y)<1e-8;}));
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
    // A farther measured hazard outside the short observation prefix still
    // rules out this early fallback; unknown space cannot conceal it.
    Eigen::Array3i index;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(2.5,.1,0.),index));
    n->map_.at("collision_risk",index)=1;n->rebuildCostmapLocked();
    EXPECT_TRUE(n->knownHazardAheadLocked({0,0,0,0},3.));
    ASSERT_TRUE(n->planPath("farther known hazard requires directed bypass"));
    EXPECT_TRUE(std::any_of(n->path_.begin(),n->path_.end(),[](const auto &p){return std::abs(p.y)>.05;}));
  }

  static void continuous_arc_keeps_analytic_heading_and_directional_certificate()
  {
    for (double side:{-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      const Pose2 start{.195,0,side*.17,0};
      const double curvature=side*.7;
      const Pose2 goal=advanceArc(start,curvature,2.2);
      std::vector<Pose2> path;
      ASSERT_TRUE(n->buildArcDirectPathLocked(start,goal,path));
      ASSERT_TRUE(n->bodyArcPathFeasible(path));
      for(size_t i=0;i<path.size();++i) {
        const auto expected=advanceArc(start,curvature,2.2*i/(path.size()-1));
        EXPECT_NEAR(wrapAngle(path[i].yaw-expected.yaw),0,1e-12);
      }
      const auto original=path;
      double score=0;
      const double distance=std::hypot(goal.x-start.x,goal.y-start.y);
      const double bearing=std::atan2(goal.y-start.y,goal.x-start.x);
      ASSERT_TRUE(n->evaluateContinuousCurveLocked(path,start,goal,distance,bearing,score));
      for(size_t i=0;i<path.size();++i) EXPECT_DOUBLE_EQ(path[i].yaw,original[i].yaw);
      auto wrong_start=start;wrong_start.yaw+=.1;
      EXPECT_FALSE(n->evaluateContinuousCurveLocked(path,wrong_start,goal,distance,bearing,score));
      auto broken=path;broken[1].yaw+=.2;
      EXPECT_FALSE(n->evaluateContinuousCurveLocked(broken,start,goal,distance,bearing,score));

      for(auto &slice:n->map_["failure_probability_upper"])slice.setOnes();
      for(double s=0;s<path.back().s+.001;s+=.001) {
        const auto p=n->samplePathAt(path,std::min(s,path.back().s));
        Eigen::Array3i index;
        ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(p.x,p.y,p.yaw),index));
        n->map_.at("failure_probability_upper",index)=1e-8F;
      }
      EXPECT_TRUE(n->chancePathSafe(path,nullptr,n->chance_path_budget_));
      EXPECT_TRUE(n->pathFootprintSafeLocked(path,.025));
      auto fake=path;n->updateYawFromTangent(fake);
      EXPECT_FALSE(n->chancePathSafe(fake,nullptr,n->chance_path_budget_));
      std::vector<Pose2> certified;
      ASSERT_TRUE(n->buildArcDirectPathLocked(start,goal,certified));
      EXPECT_NEAR(certified.front().yaw,start.yaw,1e-12);
    }
  }

  static void continuous_dubins_preserves_both_boundary_headings()
  {
    for(double side:{-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      const Pose2 start{0,0,side*.17,0};
      TerrainNavNode::DubinsCandidate candidate;
      candidate.types={side>0?'L':'R','S',side>0?'L':'R'};
      candidate.params={.35,.7,.3};candidate.length=1.35;
      auto goal=advanceArc(start,side*.5,.7);
      goal=advanceArc(goal,0,1.4);goal=advanceArc(goal,side*.5,.6);
      std::vector<Pose2> path;
      ASSERT_TRUE(n->sampleDubinsLocked(start,goal,candidate,2.,path));
      EXPECT_NEAR(path.front().yaw,start.yaw,1e-12);
      EXPECT_NEAR(wrapAngle(path.back().yaw-goal.yaw),0,1e-12);
      EXPECT_TRUE(n->bodyArcPathFeasible(path));
      ASSERT_TRUE(n->buildDubinsDirectPathLocked(start,goal,path));
      EXPECT_NEAR(path.front().yaw,start.yaw,1e-12);
      EXPECT_NEAR(wrapAngle(path.back().yaw-goal.yaw),0,1e-12);
      EXPECT_TRUE(n->bodyArcPathFeasible(path));
      EXPECT_TRUE(n->chancePathSafe(path,nullptr,n->chance_path_budget_));
    }
  }

  static void continuous_frenet_certifies_analytic_boundary_headings()
  {
    for(double side:{-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      const Pose2 start{0,0,side*.12,0},goal{2.8,side*.65,side*.35,0};
      std::vector<Pose2> path;
      ASSERT_TRUE(n->buildFrenetDirectPathLocked(start,goal,path));
      EXPECT_NEAR(path.front().yaw,start.yaw,1e-12);
      EXPECT_NEAR(path.back().yaw,goal.yaw,1e-12);
      EXPECT_TRUE(n->bodyArcPathFeasible(path));
      EXPECT_TRUE(n->chancePathSafe(path,nullptr,n->chance_path_budget_));
      auto impossible=goal;impossible.yaw=M_PI;
      EXPECT_FALSE(n->buildFrenetDirectPathLocked(start,impossible,path));
    }
  }

  static void continuous_replan_keeps_curved_committed_prefix()
  {
    for(double side:{-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      n->path_commit_enable_=true;n->path_commit_distance_=.5;n->path_commit_min_distance_=.2;
      const Pose2 start{0,0,side*.17,0};
      n->odom_.valid=true;n->odom_.yaw=start.yaw;
      n->have_goal_=true;n->raw_goal_=advanceArc(start,side*.5,2.5);
      n->have_path_=true;
      for(int i=0;i<=15;++i) n->path_.push_back(advanceArc(start,side*.5,i*.05));
      n->updatePathArcLength(n->path_);
      std::vector<Pose2> prefix;Pose2 search_start;
      ASSERT_TRUE(n->buildCommittedPrefixLocked(n->odom_,n->path_,prefix,search_start));
      ASSERT_TRUE(n->planPath("rolling local goal should advance"));
      ASSERT_GT(n->path_.size(),prefix.size());
      for(size_t i=0;i<prefix.size();++i) {
        EXPECT_DOUBLE_EQ(n->path_[i].x,prefix[i].x);
        EXPECT_DOUBLE_EQ(n->path_[i].y,prefix[i].y);
        EXPECT_DOUBLE_EQ(n->path_[i].yaw,prefix[i].yaw);
      }
      EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
      EXPECT_TRUE(n->pathFootprintSafeLocked(n->path_,.025));
      EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
    }
  }

  static void observation_fallback_keeps_committed_connector()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->chance_allow_startup_prior_=true;
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    n->map_.add("probability_pose_event",true,1);
    n->path_commit_enable_=true;n->path_commit_distance_=.4;n->path_commit_min_distance_=.2;
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_={3.,0.,0.,0.};
    n->have_path_=true;n->path_={{0,0,0,0},{.1,0,0,.1},{.2,0,0,.2},{.3,0,0,.3},{.4,0,0,.4}};
    for(int r=0;r<40;++r) for(int c=0;c<40;++c) {
      Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
      if(std::hypot(p.x(),p.y())>.9)for(auto &slice:n->map_["probability_valid"])slice(r,c)=0;
    }
    n->rebuildCostmapLocked();
    ASSERT_TRUE(n->planPath("rolling local goal should advance"));
    ASSERT_GT(n->path_.size(),2u);
    EXPECT_NEAR(n->path_.front().x,0.,1e-9);
    EXPECT_GT(n->path_.back().x,.5);EXPECT_LT(n->path_.back().x,.91);
    EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
  }

  static void certified_turn_precedes_observation_only_progress()
  {
    for (double side : {-1., 1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      n->chance_allow_startup_prior_=true;
      ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
      n->map_.add("probability_pose_event",true,1);
      n->odom_.valid=true;n->have_goal_=true;n->raw_goal_={2.,side*.7,0.,0.};
      ASSERT_GT(n->observationPrefixLimit({0,0,0,0},n->raw_goal_),.25);
      ASSERT_TRUE(n->planPath("certified turn before observation fallback"));
      ASSERT_GT(n->path_.size(),2u);
      EXPECT_GT(side*n->path_.back().y,.4);
      EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
      EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
      for (auto &slice:n->map_["probability_valid"])slice.setZero();
      EXPECT_FALSE(n->planPath("unknown still cannot take the turn"));
    }
  }

  static void guide_continuity_is_soft_and_never_admits_blocked_side()
  {
    for (double side : {-1.,1.}) {
      auto n=node(1e-8F);n->live_obstacle_enable_=false;
      n->guide_follow_weight_=6.;n->local_goal_distance_=3.;
      n->guide_min_distance_=1.;n->max_search_time_ms_=300.;
      for(int r=0;r<40;++r) for(int c=0;c<40;++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
        if(p.x()>1.2 && p.x()<2.2 && std::abs(p.y())<.6)
          n->map_.at("collision_risk",Eigen::Array3i(r,c,0))=1;
      }
      n->rebuildCostmapLocked();
      const std::vector<Pose2> previous={{0,0,0,0},{.5,side*.2,side*.4,.54},
                                        {1.,side*.7,side*.5,1.25}};
      OdomState odom;odom.valid=true;
      const Pose2 raw{3.5,0,0,0};Pose2 active;std::string reason;std::vector<Pose2> guide;
      ASSERT_TRUE(n->resolveGuideGoalLocked(odom,raw,active,reason,guide,previous));
      ASSERT_FALSE(guide.empty());
      EXPECT_TRUE(std::any_of(guide.begin(),guide.end(),[&](const auto &p){return side*p.y>.6;}));
      EXPECT_TRUE(std::all_of(guide.begin(),guide.end(),[&](const auto &p){return side*p.y>=-.2;}));
      // New occupancy on that whole side must override the preference.
      for(int r=0;r<40;++r) for(int c=0;c<40;++c) {
        Eigen::Vector3d p;n->map_.index2Pos(Eigen::Array3i(r,c,0),p);
        if(side*p.y()>.3)n->map_.at("collision_risk",Eigen::Array3i(r,c,0))=1;
      }
      n->rebuildCostmapLocked();
      ASSERT_TRUE(n->resolveGuideGoalLocked(odom,raw,active,reason,guide,previous));
      EXPECT_TRUE(std::any_of(guide.begin(),guide.end(),[&](const auto &p){return side*p.y<-.6;}));
      double cost=0;EXPECT_TRUE(n->footprintCostLocked(active.x,active.y,active.yaw,cost));
    }
  }

  static void inflated_live_body_is_not_a_second_tracking_envelope()
  {
    auto n=node(.001F);
    n->footprint_front_=n->footprint_back_=.42;n->footprint_half_width_=.313;
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.4;
    n->footprint_margin_=.1;n->live_obstacle_inflation_radius_=.25;
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->min_x=live->min_y=-4.;live->resolution=.2;
    live->width=live->height=40;live->hard.assign(1600,0);
    live->hard[18*40+31]=1; // Recorded inflated cell (2.3,-.3).
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    double cost=0;
    EXPECT_TRUE(n->footprintCostLocked(1.584095666,-.625967177,-.413341027,cost));
    const std::vector<Pose2> escaped={{1.584095666,-.625967177,-.413341027,0.}};
    EXPECT_FALSE(n->livePathAheadUnsafe(escaped,0));
    // The same cell as uninflated terrain occupancy retains the extra margin.
    Eigen::Array3i index;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(2.3,-.3,0),index));
    n->map_["collision_risk"][0](index[0],index[1])=1;
    EXPECT_FALSE(n->footprintCostLocked(1.584095666,-.625967177,-.413341027,cost));
    n->map_["collision_risk"][0](index[0],index[1])=0;
    // Actual bumper overlap cannot be excused by the layer separation.
    EXPECT_FALSE(n->footprintCostLocked(1.9,-.3,0,cost));
    // No inflation means the full configured body margin is still required.
    n->live_obstacle_inflation_radius_=0;
    EXPECT_FALSE(n->footprintCostLocked(1.584095666,-.625967177,-.413341027,cost));
  }

  static void grid_cell_overlap_uses_all_separating_axes()
  {
    EXPECT_TRUE(bodyOverlapsCell(0,0,0,.58,.48,.4,.6,0,.2));
    EXPECT_FALSE(bodyOverlapsCell(0,0,0,.58,.48,.4,.8,0,.2));
    EXPECT_TRUE(bodyOverlapsCell(0,0,M_PI/2,.58,.48,.4,0,.6,.2));
    // Inside both local-axis projection bounds, outside the world-X bound.
    EXPECT_FALSE(bodyOverlapsCell(0,0,M_PI/4,.5,.5,.5,.85,0,.2));
  }

  static void guide_search_uses_snapshot_without_holding_control_goal_lock()
  {
    auto n=node(1e-9F);
    n->have_goal_=true;n->raw_goal_={100.,100.,0.,0.};
    OdomState odom;odom.valid=true;odom.x=-.8;
    const Pose2 captured_goal{.8,0.,0.,0.};
    Pose2 active;std::vector<Pose2> guide;std::string reason;
    std::promise<bool> result;auto future=result.get_future();
    std::unique_lock<std::mutex> goal_lock(n->goal_mutex_);
    std::thread search([&] {
      result.set_value(n->resolveActiveGoalLocked(odom,captured_goal,active,reason,guide));
    });
    const auto status=future.wait_for(std::chrono::seconds(1));
    goal_lock.unlock();search.join();
    EXPECT_EQ(status,std::future_status::ready);
    ASSERT_TRUE(future.get());
    EXPECT_NEAR(active.x,captured_goal.x,1e-9);
    EXPECT_NEAR(active.y,captured_goal.y,1e-9);
  }

  static std::unique_ptr<TerrainNavNode> node(float probability)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("chance_constrained",true)});
    auto n=std::make_unique<TerrainNavNode>(options,false);
    n->map_.initGeometry(Eigen::Array2d(8,8),.2,2*M_PI/36,Eigen::Vector2d::Zero(),2*M_PI);
    for (const auto &entry : std::vector<std::pair<std::string,float>>{
      {"probability_contract",1}, {"probability_front",.52F}, {"probability_back",.52F},
      {"probability_half_width",.413F}, {"probability_max_pitch",.55F},
      {"probability_max_roll",.45F}, {"probability_max_step",.18F},
      {"probability_gp_enabled",0}, {"collision_risk",0}})
      n->map_.add(entry.first,false,entry.second);
    n->map_.add("failure_probability_upper",true,probability);
    n->map_.add("probability_valid",true,1);
    n->rebuildCostmapLocked();
    n->map_ready_=true;
    auto snapshot=std::make_shared<ControlMapSnapshot>();
    snapshot->grid=n->map_; snapshot->stamp=n->now();
    snapshot->min_x=n->min_x_; snapshot->min_y=n->min_y_; snapshot->resolution=n->resolution_;
    snapshot->width=n->width_; snapshot->height=n->height_; snapshot->costs=n->costs_;
    snapshot->clearance=n->clearance_; snapshot->comfort_clearance=n->comfort_clearance_;
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    return n;
  }

  static void contract()
  {
    auto n=node(.001F);
    ASSERT_TRUE(n->probabilityContractValid(n->map_));
    n->map_["probability_gp_enabled"][0].setConstant(1);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    n->chance_allow_gp_completion_=true;
    EXPECT_TRUE(n->probabilityContractValid(n->map_));
    n->map_["probability_front"][0].setConstant(.1F);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
  }
  static void clearance_span_contract_is_separate_from_local_step_and_fails_closed()
  {
    auto n=node(.0001F);n->set_parameter(rclcpp::Parameter("use_sim_time",true));
    n->chance_allow_joint_model_=true;n->chance_max_step_m_=.1;
    auto &g=n->map_;g["probability_contract"][0].setConstant(7.F);
    g["probability_max_step"][0].setConstant(.1F);
    EXPECT_FALSE(n->probabilityContractValid(g)); // metadata absent
    for(const auto &entry:std::vector<std::pair<std::string,float>>{
      {"probability_startup_prior_enabled",0},{"probability_joint_model",1},{"probability_step_model",1},
      {"probability_height_radius_cells",8},{"probability_max_roughness",.125F},
      {"probability_ground_clearance",.135F},{"probability_underbody_margin",.01F}})g.add(entry.first,false,entry.second);
    g.add("height_error_tail_upper",true,.0001F);g.add("probability_pose_event",true,1.F);
    ASSERT_TRUE(n->probabilityContractValid(g));
    g["probability_max_roughness"][0].setConstant(.126F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_max_roughness"][0].setConstant(.125F);
    g["probability_ground_clearance"][0].setConstant(.2F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_ground_clearance"][0].setConstant(.135F);
    g["probability_underbody_margin"][0].setConstant(0.F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_underbody_margin"][0].setConstant(.01F);
    g["probability_max_step"][0].setConstant(.11F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_max_step"][0].setConstant(.1F);
    n->chance_allow_joint_model_=false;EXPECT_FALSE(n->probabilityContractValid(g));
    n->chance_allow_joint_model_=true;EXPECT_TRUE(n->probabilityContractValid(g));
    // A different covariance prior is never silently accepted as contract 7.
    g["probability_contract"][0].setConstant(8.F);
    EXPECT_FALSE(n->probabilityContractValid(g));
    n->chance_allow_independent_model_floor_=true;EXPECT_FALSE(n->probabilityContractValid(g));
    g.add("probability_floor_covariance_model",false,1.F);EXPECT_TRUE(n->probabilityContractValid(g));
    g["probability_floor_covariance_model"][0].setConstant(0.F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_floor_covariance_model"][0].setConstant(1.F);
    g["probability_contract"][0].setConstant(7.F);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_contract"][0].setConstant(8.F);
    n->set_parameter(rclcpp::Parameter("use_sim_time",false));EXPECT_FALSE(n->probabilityContractValid(g));
    n->set_parameter(rclcpp::Parameter("use_sim_time",true));
    for(auto &s:g["probability_valid"])s.setZero();
    EXPECT_FALSE(n->chancePathSafe({{0,0,0,0},{.3,0,0,.3}},nullptr,.1));
  }

  static void bounded_mean_contract_requires_permission_and_only_reuses_pose_events()
  {
    auto n=node(.04F);n->set_parameter(rclcpp::Parameter("use_sim_time",true));
    n->chance_allow_joint_model_=true;n->chance_allow_independent_model_floor_=true;
    n->chance_max_step_m_=.1;
    auto &g=n->map_;g["probability_contract"][0].setConstant(9.F);
    g["probability_max_step"][0].setConstant(.1F);
    for(const auto &entry:std::vector<std::pair<std::string,float>>{
      {"probability_startup_prior_enabled",0},{"probability_joint_model",1},{"probability_step_model",1},
      {"probability_floor_covariance_model",1},{"probability_height_radius_cells",8},
      {"probability_max_roughness",.125F},{"probability_ground_clearance",.135F},{"probability_underbody_margin",.01F},
      {"probability_bounded_pending_model",1},{"probability_bounded_pending_sources",2},
      {"bounded_pending_used",0},{"bounded_mean_radius_m",.02F},{"bounded_mean_lower",-.02F},{"bounded_mean_upper",.02F},
      {"probability_variance",1e-20F},{"probability_height_std_floor",0},{"probability_variance_scale",1}})
      g.add(entry.first,false,entry.second);
    g.add("height_error_tail_upper",true,1.F);g.add("probability_pose_event",true,1.F);
    g.add("height_error_radius_m",true,.1F);
    EXPECT_FALSE(n->probabilityContractValid(g));
    n->chance_allow_bounded_pending_=true;ASSERT_TRUE(n->probabilityContractValid(g));
    // Tiny Gaussian variance must not erase the bounded-mean family through
    // the legacy absolute-height alternative certificate.
    const auto q=n->queryCostLocked(0,0,0);
    ASSERT_TRUE(q.known);EXPECT_FALSE(q.lethal);EXPECT_EQ(q.height_radius,0);
    EXPECT_TRUE(n->chancePathSafe({{0,0,0,0}},nullptr,.1));
    EXPECT_FALSE(n->chancePathSafe({{0,0,0,0},{1,0,0,1}},nullptr,.1));
    for(auto &s:g["probability_pose_event"])s.setZero();
    EXPECT_TRUE(n->queryCostLocked(0,0,0).lethal);
    for(auto &s:g["probability_pose_event"])s.setOnes();
    g["probability_bounded_pending_sources"][0].setZero();EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_bounded_pending_sources"][0].setConstant(129);EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_bounded_pending_sources"][0].setConstant(2);
    g["probability_bounded_pending_model"][0].setZero();EXPECT_FALSE(n->probabilityContractValid(g));
    g["probability_bounded_pending_model"][0].setOnes();
    n->set_parameter(rclcpp::Parameter("use_sim_time",false));EXPECT_FALSE(n->probabilityContractValid(g));
  }

  static void joint_budget()
  {
    auto n=node(.04F);
    EXPECT_FALSE(n->queryCostLocked(0,0,0).lethal);
    const std::vector<Pose2> path={{0,0,0,0},{1,0,0,1}};
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["failure_probability_upper"][0].setConstant(.001F);
    // Yaw zero queries slice zero. The safety check must see float data, not
    // the old uint8 grid (which has not been rebuilt after changing p).
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
  }

  static void unknown_cannot_use_legacy_bubble()
  {
    auto n=node(.001F);
    n->unknown_robot_pose_ready_=true;
    n->unknown_robot_x_=n->unknown_robot_y_=0;
    n->unknown_robot_bubble_radius_=10;
    n->map_["probability_valid"][0].setZero();
    EXPECT_TRUE(n->queryCostLocked(0,0,0,true).lethal);
    n->map_["probability_valid"][0].setOnes();
    n->map_["failure_probability_upper"][0].setConstant(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(n->queryCostLocked(0,0,0,true).lethal);
  }

  static void one_cell_collision_is_not_a_probability_average()
  {
    auto n=node(.001F);
    Eigen::Array3i index;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.3,.1,0),index));
    n->map_["collision_risk"][0](index[0],index[1])=1;
    EXPECT_TRUE(n->queryCostLocked(.3,.1,0).known);
    EXPECT_TRUE(n->queryCostLocked(.3,.1,0).lethal);
    double cost=0;
    EXPECT_FALSE(n->footprintCostLocked(0,0,.15,cost));
  }

  static void live_columns_require_measured_fresh_ground()
  {
    for(int mode=0;mode<4;++mode) {
      auto n=node(.001F);
      n->live_obstacle_min_hits_=2;
      auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
      snapshot->grid.add("terrain_observed",false,mode==1?0.F:1.F);
      snapshot->grid.add("gp_elevation_observed",false,
        mode==3?std::numeric_limits<float>::quiet_NaN():0.F);
      snapshot->grid.add("elevation",false,0.F);
      if(mode==2)snapshot->stamp=n->now()-rclcpp::Duration::from_seconds(1.);
      std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
      auto cloud=std::make_shared<sensor_msgs::msg::PointCloud2>();
      cloud->header.frame_id="map";cloud->header.stamp=n->now();
      sensor_msgs::PointCloud2Modifier modifier(*cloud);
      modifier.setPointCloud2FieldsByString(1,"xyz");modifier.resize(2);
      sensor_msgs::PointCloud2Iterator<float> x(*cloud,"x"),y(*cloud,"y"),z(*cloud,"z");
      *x=-1.1F;*y=-1.3F;*z=.6F;++x;++y;++z;
      *x=-.9F;*y=-1.3F;*z=.6F;
      n->cloudCallback(cloud);
      const auto live=std::atomic_load(&n->live_snapshot_);
      ASSERT_TRUE(live);
      const auto occupied=std::count(live->hard.begin(),live->hard.end(),1);
      if(mode==0)EXPECT_GT(occupied,0);else EXPECT_EQ(occupied,0);
    }
  }

  static void raised_bumper_does_not_shrink_to_ground_contact_stencil()
  {
    auto n=node(.001F);double cost=0;
    Eigen::Array3i index;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(.7,.1,0),index));
    n->map_["collision_risk"][0](index[0],index[1])=1;
    EXPECT_TRUE(n->footprintCostLocked(0,0,0,cost));
    n->body_collision_front_=.58;n->body_collision_back_=.48;n->body_collision_half_width_=.40;
    EXPECT_FALSE(n->footprintCostLocked(0,0,0,cost));
    // The producer's ground-contact probability geometry is still compatible.
    EXPECT_TRUE(n->probabilityContractValid(n->map_));
  }

  static void sampled_control_fails_closed()
  {
    auto n=node(.9F);
    OdomState odom; odom.valid=true;
    const std::vector<Pose2> path={{0,0,0,0},{1,0,0,1}};
    const auto command=n->computeSamplingMpcCommand(odom,path);
    EXPECT_FALSE(n->sampling_feasible_);
    EXPECT_EQ(command.linear.x,0);
  }

  static void sampled_control_does_not_raise_a_safe_speed()
  {
    auto n=node(1e-9F);
    n->mppi_batch_size_=2; n->mppi_steps_=6; n->min_tracking_speed_=1.;
    OdomState odom; odom.valid=true;
    const std::vector<Pose2> path={{0,0,0,0},{2,0,0,2}};
    const auto command=n->computeSamplingMpcCommand(odom,path);
    ASSERT_TRUE(n->sampling_feasible_);
    EXPECT_LE(command.linear.x,n->max_accel_*n->mppi_dt_+1e-9);
  }

  static void controller_names_are_not_mppi()
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("tracking_controller","mppi")});
    EXPECT_THROW(TerrainNavNode(options,false),std::invalid_argument);
  }

  static void goal_runs_complete_planner()
  {
    auto n=node(.001F);
    n->odom_.valid=true;
    n->odom_.yaw=-.001;
    n->have_goal_=true;
    n->raw_goal_=Pose2{2,-.002,-.001,0};
    n->live_obstacle_enable_=false;
    EXPECT_TRUE(n->planPath("qualification goal"));
    EXPECT_GE(n->path_.size(),2u);
  }

  static void short_terminal_connector_is_not_a_failed_frontier()
  {
    auto n=node(1e-6F);n->goal_tolerance_=.22;n->goal_reached_distance_=.25;
    const Pose2 start{0,0,0,0},goal{.55,.08,0,0};
    std::vector<Pose2> close={{0,0,0,0},{.14,.01,0,0},{.28,.03,0,0},{.42,.05,0,0}};
    EXPECT_TRUE(n->latticePathReasonableLocked(close,start,goal,false));
    EXPECT_FALSE(n->latticePathReasonableLocked(close,start,goal,true));
    EXPECT_FALSE(n->latticePathReasonableLocked(close,start,Pose2{1.5,1,0,0},false));
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=goal;n->live_obstacle_enable_=false;
    ASSERT_TRUE(n->planPath("short terminal regression"));
    ASSERT_FALSE(n->path_.empty());
    EXPECT_LE(std::hypot(n->path_.back().x-goal.x,n->path_.back().y-goal.y),n->goal_tolerance_);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
  }

  static void off_axis_observation_retains_turning_room()
  {
    auto n=node(.001F);n->max_steer_=.4;
    const Pose2 start{0,0,0,0},goal{2,.75,0,0};
    const double length=n->observationPrefixLimit(start,goal);
    ASSERT_GE(length,.25);ASSERT_LE(length,.4);
    EXPECT_LE(1.5/(std::pow(2-length,2)+.75*.75),.8*n->maxReferenceCurvature()+1e-9);
    EXPECT_LT(n->observationPrefixLimit(Pose2{.4,0,0,0},goal),.25);
    EXPECT_EQ(n->observationPrefixLimit(start,Pose2{.5,.5,0,0}),0);
    EXPECT_EQ(n->observationPrefixLimit(start,Pose2{-.5,.5,0,0}),0);
    EXPECT_DOUBLE_EQ(n->observationPrefixLimit(start,Pose2{2,0,0,0}),1.2);
  }

  static void certified_short_lattice_frontier_is_useful_progress()
  {
    auto n=node(1e-7F);
    const Pose2 start{0,0,0,0},goal{3,0,0,0};
    std::vector<Pose2> path={{0,0,0,0},{.28,0,0,.28},{.56,0,0,.56},{.84,0,0,.84}};
    ASSERT_TRUE(n->chancePathSafe(path,nullptr,n->chance_path_budget_));
    EXPECT_TRUE(n->latticePathReasonableLocked(path,start,goal,true));
    n->chance_constrained_=false;
    EXPECT_FALSE(n->latticePathReasonableLocked(path,start,goal,true));
    n->chance_constrained_=true;
    for(int k=0;k<n->map_.getSizeYaw();++k) {
      for(int i=0;i<n->map_.getSizePos()[0];++i)for(int j=0;j<n->map_.getSizePos()[1];++j) {
        Eigen::Vector3d p;
        ASSERT_TRUE(n->map_.index2Pos(Eigen::Array3i(i,j,k),p));
        if(p.x()>1.6)n->map_["probability_valid"][k](i,j)=0;
      }
    }
    Eigen::Array3i obstacle;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(2.1,.1,0),obstacle));
    n->map_["collision_risk"][0](obstacle[0],obstacle[1])=1;
    n->rebuildCostmapLocked();
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=goal;n->live_obstacle_enable_=false;
    n->direct_curve_enable_=false;n->max_iterations_=8;
    ASSERT_TRUE(n->planPath("short certified lattice frontier"));
    ASSERT_GE(n->path_.size(),2u);
    EXPECT_GE(n->path_.back().s,.5);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
    EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
    EXPECT_GT(std::hypot(n->path_.back().x-goal.x,n->path_.back().y-goal.y),n->goal_tolerance_);
    EXPECT_EQ(n->observation_path_seq_,n->path_seq_);
  }

  static void mission_progress_fan_turns_after_failed_guide_without_relaxing_safety()
  {
    auto n=node(1e-8F);n->max_steer_=.4;
    const Pose2 start{0,0,.75,0},goal{6,0,0,0};std::vector<Pose2> path;
    EXPECT_FALSE(n->buildCertifiedForwardPrefixLocked(start,goal,path)); // Old bearing cutoff.
    ASSERT_TRUE(n->buildMissionProgressFanLocked(start,goal,path));
    ASSERT_GE(path.back().s,.8);EXPECT_LE(path.back().s,1.2+1e-9);
    EXPECT_LT(std::hypot(goal.x-path.back().x,goal.y-path.back().y),5.55);
    EXPECT_LT(std::abs(wrapAngle(std::atan2(goal.y-path.back().y,goal.x-path.back().x)-path.back().yaw)),.65);
    EXPECT_TRUE(n->bodyArcPathFeasible(path));EXPECT_TRUE(n->pathFootprintSafeLocked(path,.025));
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    EXPECT_FALSE(n->buildMissionProgressFanLocked(start,Pose2{-6,0,0,0},path));
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_FALSE(n->buildMissionProgressFanLocked(start,goal,path));
    n=node(1e-8F);n->map_["collision_risk"][0].setOnes();n->rebuildCostmapLocked();
    EXPECT_FALSE(n->buildMissionProgressFanLocked(start,goal,path));
    n=node(.04F);
    EXPECT_FALSE(n->buildMissionProgressFanLocked(start,goal,path));
  }

  static void planar_layers_ignore_heading()
  {
    auto n=node(.001F);
    const auto &constant_grid=n->map_;
    for (double yaw : {-6.1,-.001,0.,.5,3.,6.2}) {
      Eigen::Array3i index;
      ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(0,0,yaw),index));
      EXPECT_EQ(n->map_.at("collision_risk",index),0);
      EXPECT_EQ(constant_grid.at("collision_risk",index),0);
      EXPECT_FLOAT_EQ(constant_grid.getValue("probability_front",Eigen::Vector3d(0,0,yaw),
          se2_grid::InterpolationMethods::INTER_NEAREST),.52F);
      EXPECT_FALSE(n->queryCostLocked(0,0,yaw).lethal);
    }
  }

  static void unknown_goal_rejects_before_search()
  {
    auto n=node(.001F);
    n->map_["probability_valid"][0].setZero();
    n->odom_.valid=true; n->have_goal_=true;
    n->raw_goal_=Pose2{2,0,0,0};
    EXPECT_FALSE(n->planPath("unknown start qualification"));
    EXPECT_TRUE(n->path_.empty());
  }

  static void shared_source_budget_reuses_ground_not_pose_events()
  {
    auto n=node(.0196128F);
    const std::vector<Pose2> path={{0,0,0,0},{2,0,0,2}};
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["probability_contract"][0].setConstant(2);
    n->map_.add("height_error_tail_upper",true,.00022852F);
    n->map_.add("probability_height_radius_cells",false,6);
    ASSERT_TRUE(n->probabilityContractValid(n->map_));
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.01));
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{2,0,0,0};
    EXPECT_TRUE(n->planPath("shared-height qualification goal"));
    const std::vector<Pose2> repeated={{0,0,0,0},{0,0,0,0}};
    EXPECT_TRUE(n->chancePathSafe(repeated,nullptr,.04));
    n->map_["probability_height_radius_cells"][0].setConstant(1);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
  }

  static void startup_assumption_requires_explicit_simulation_permission()
  {
    auto n=node(.001F);
    n->map_["probability_contract"][0].setConstant(4);
    n->map_.add("probability_pose_event",true,1);
    n->map_.add("height_error_tail_upper",true,1e-6F);
    n->map_.add("probability_height_radius_cells",false,4);
    n->map_.add("probability_startup_prior_enabled",false,1);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    n->chance_allow_startup_prior_=true;
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    EXPECT_TRUE(n->probabilityContractValid(n->map_));
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{2,0,0,0};
    EXPECT_TRUE(n->planPath("simulation conditional support"));
    n->map_["failure_probability_upper"][0].setConstant(.02);
    n->chance_path_budget_=.10;
    const std::vector<Pose2> long_path={{0,0,0,0},{2,0,0,2}};
    EXPECT_FALSE(n->chancePathSafe(long_path,nullptr,.1));
    n->path_.clear();n->have_path_=false;
    EXPECT_TRUE(n->planPath("bounded conditional prefix"));
    ASSERT_FALSE(n->path_.empty());
    EXPECT_GE(n->path_.back().s,.25);EXPECT_LT(n->path_.back().s,2.);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,n->chance_path_budget_));
    EXPECT_EQ(n->raw_goal_.x,2.);
  }

  static void shared_gp_requires_permission_and_preserves_navigation_checks()
  {
    auto n=node(1e-6F);
    n->map_["probability_contract"][0].setConstant(5);
    n->map_.add("probability_pose_event",true,1);
    n->map_.add("height_error_tail_upper",true,.1F);
    n->map_.add("probability_height_radius_cells",false,4);
    n->map_.add("probability_startup_prior_enabled",false,0);
    n->map_.add("probability_joint_model",false,1);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    n->chance_allow_joint_model_=true;
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    ASSERT_TRUE(n->probabilityContractValid(n->map_));
    const std::vector<Pose2> path={{0,0,0,0},{2,0,0,2}};
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    n->map_["failure_probability_upper"][0].setConstant(.04F);
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["failure_probability_upper"][0].setConstant(1e-6F);
    n->map_["probability_valid"][0].setZero();
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["probability_valid"][0].setOnes();
    n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["collision_risk"][0].setZero();
    n->map_["probability_contract"][0].setConstant(6);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    n->map_.add("probability_step_model",false,0);
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
    n->map_["probability_step_model"][0].setOnes();
    EXPECT_TRUE(n->probabilityContractValid(n->map_));
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    n->map_["probability_joint_model"][0].setZero();
    EXPECT_FALSE(n->probabilityContractValid(n->map_));
  }

  static void reverse_progresses_only_on_certified_prefix()
  {
    auto n=node(.02F);
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{-2,0,0,0};
    n->live_obstacle_enable_=false;
    const std::vector<Pose2> whole={{0,0,0,0,-1},{-2,0,0,2,-1}};
    ASSERT_FALSE(n->chancePathSafe(whole,nullptr,.1));
    ASSERT_TRUE(n->planPath("certified rolling reverse"));
    ASSERT_GE(n->path_.size(),2u);
    EXPECT_GE(n->path_.back().s,.25);EXPECT_LT(n->path_.back().s,2.);
    EXPECT_EQ(n->path_.back().direction,-1);
    EXPECT_EQ(n->raw_goal_.x,-2.);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    n->map_["probability_valid"][0].setZero();
    EXPECT_FALSE(n->planPath("unknown reverse must not progress"));
  }

  static void offset_reverse_goal_uses_certified_terminal_disk()
  {
    auto n=node(1e-6F);
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{-.84,.4,0,0};
    n->goal_tolerance_=.22;n->live_obstacle_enable_=false;
    ASSERT_TRUE(n->planPath("offset reverse terminal disk"));
    ASSERT_GT(n->path_.size(),2u);
    EXPECT_EQ(n->path_.front().direction,-1);
    EXPECT_LE(std::hypot(n->path_.back().x+.84,n->path_.back().y-.4),.22);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    n->map_["probability_valid"][0].setZero();
    EXPECT_FALSE(n->planPath("offset reverse unknown must stop"));
    n->map_["probability_valid"][0].setOnes();
    n->map_["failure_probability_upper"][0].setConstant(.9F);
    EXPECT_FALSE(n->planPath("offset reverse high risk must stop"));
  }

  static void observation_prefix_does_not_override_known_hazard()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    n->chance_allow_startup_prior_=true;
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    n->map_.add("probability_pose_event",true,1);
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    EXPECT_FALSE(n->knownHazardAheadLocked(Pose2{0,0,0,0},2.));
    Eigen::Array3i obstacle;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(1.7,.1,0),obstacle));
    n->map_["collision_risk"][0](obstacle[0],obstacle[1])=1;
    n->rebuildCostmapLocked();
    ASSERT_TRUE(n->knownHazardAheadLocked(Pose2{0,0,0,0},2.));
    ASSERT_TRUE(n->planPath("known hazard must bypass observation prefix"));
    ASSERT_FALSE(n->path_.empty());
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    EXPECT_TRUE(std::any_of(n->path_.begin(),n->path_.end(),
      [](const Pose2 &p){return std::abs(p.y)>.05;}));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    n->map_["collision_risk"][0].setZero();
    EXPECT_FALSE(n->knownHazardAheadLocked(Pose2{0,0,0,0},2.));
  }

  static void lattice_keeps_continuous_arc_and_certified_body_heading()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    std::vector<TerrainNavNode::SearchNode> nodes(1);
    nodes[0].yaw=.031;
    const Pose2 start{0,0,.031,0};
    for(double steer:{.4,-.4,.132}) {
      const auto primitive=n->propagatePrimitiveLocked(nodes.back(),steer);
      ASSERT_TRUE(primitive.ok);
      TerrainNavNode::SearchNode child;
      child.x=primitive.end.x;child.y=primitive.end.y;child.yaw=primitive.end.yaw;
      child.steer=steer;child.parent=static_cast<int>(nodes.size()-1);
      child.yaw_bin=n->yawToBin(child.yaw);
      child.probability_events=primitive.probability_events;
      nodes.push_back(child);
    }
    const auto path=n->extractPath(nodes,3);
    ASSERT_GT(path.size(),20u);
    EXPECT_NEAR(path.front().yaw,start.yaw,1e-12);
    EXPECT_NEAR(path.back().x,nodes.back().x,1e-9);
    EXPECT_NEAR(path.back().y,nodes.back().y,1e-9);
    EXPECT_NEAR(path.back().yaw,nodes.back().yaw,1e-9);
    EXPECT_TRUE(n->bodyArcPathFeasible(path));
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    const double curvature=n->primitiveCurvature(.4);
    auto split=start;
    for(int i=0;i<7;++i)split=advanceArc(split,curvature,.06);
    const auto single=advanceArc(start,curvature,.42);
    EXPECT_NEAR(split.x,single.x,1e-12);EXPECT_NEAR(split.y,single.y,1e-12);
    EXPECT_NEAR(split.yaw,single.yaw,1e-12);
    nodes[1].yaw=n->binToYaw(nodes[1].yaw_bin);
    EXPECT_TRUE(n->extractPath(nodes,3).empty());
    auto invalid=path;invalid[1].x=NAN;
    EXPECT_FALSE(n->bodyArcPathFeasible(invalid));
  }

  static void terminal_two_gear_connector_preserves_support_risk_and_stopped_handover()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=true;n->live_obstacle_inflation_radius_=0.;
    n->rear_axle_offset_=.215;n->wheel_base_=.430;n->max_steer_=.31432913231137;
    n->body_collision_front_=n->body_collision_back_=.421;n->body_collision_half_width_=.360;
    n->footprint_margin_=.10;n->hard_clearance_floor_=0.;n->goal_tolerance_=.10;
    n->reverse_enable_=n->reverse_recovery_enable_=true;
    n->have_goal_=true;n->goal_seq_=1;n->raw_goal_={-.5454674958414,-.1917677776949,0.,0.};
    n->odom_.valid=true;n->odom_.stamp=n->now();n->map_stamp_=n->now();
    auto live=std::make_shared<LiveObstacleSnapshot>();live->min_x=live->min_y=-4;
    live->width=live->height=40;live->resolution=.2;live->stamp_sec=n->now().seconds();
    live->hard.assign(1600,0);live->present_hard=live->hard;
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    std::vector<Pose2> direct;
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(Pose2{},n->raw_goal_,direct));
    ASSERT_TRUE(n->planPath("recorded nearby rear goal requires a parking maneuver"));
    ASSERT_TRUE(n->recovery_staged_gears_);ASSERT_EQ(n->recovery_legs_.size(),1u);
    ASSERT_FALSE(n->path_.empty());EXPECT_EQ(n->path_.front().direction,1);
    EXPECT_EQ(n->recovery_legs_.front().front().direction,-1);
    const auto target=n->recovery_legs_.front().back();
    EXPECT_LE(std::hypot(target.x-n->raw_goal_.x,target.y-n->raw_goal_.y),n->goal_tolerance_);
    auto full=n->path_;full.insert(full.end(),n->recovery_legs_.front().begin(),n->recovery_legs_.front().end());
    EXPECT_TRUE(n->chancePathSafe(full,nullptr,n->chance_path_budget_));
    auto odom=n->odom_;odom.x=n->path_.back().x;odom.y=n->path_.back().y;odom.yaw=n->path_.back().yaw;
    odom.v=.1;
    EXPECT_FALSE(n->activateNextStagedGearLocked(odom,1,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,1);
    odom.v=0.;ASSERT_TRUE(n->activateNextStagedGearLocked(odom,1,n->input_epoch_.load()));
    EXPECT_EQ(n->path_.front().direction,-1);
    std::vector<std::vector<Pose2>> legs;
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->buildCertifiedTerminalManeuverLocked(Pose2{},n->raw_goal_,legs));
    for(auto &slice:n->map_["probability_valid"])slice.setOnes();
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(.9F);
    EXPECT_FALSE(n->buildCertifiedTerminalManeuverLocked(Pose2{},n->raw_goal_,legs));
    for(auto &slice:n->map_["failure_probability_upper"])slice.setConstant(1e-8F);
    n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->buildCertifiedTerminalManeuverLocked(Pose2{},n->raw_goal_,legs));
  }

  static void terminal_reverse_checks_alternative_arcs_before_stopping()
  {
    auto n=node(1e-6F);n->goal_tolerance_=.22;
    const Pose2 start{.0685,.169,.214,0,-1},goal{-.1846,.0147,0,0,-1};
    Eigen::Array3i blocked;
    ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(-.03,.15,.214),blocked));
    n->map_["probability_valid"][blocked[2]](blocked[0],blocked[1])=0;
    std::vector<Pose2> path;
    ASSERT_TRUE(n->buildCertifiedReverseArcLocked(start,goal,path));
    EXPECT_LT(path.back().yaw,start.yaw);
    EXPECT_LE(std::hypot(path.back().x-goal.x,path.back().y-goal.y),.22);
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    for(auto &slice:n->map_["probability_valid"])slice.setZero();
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,goal,path));
  }

  static void conditional_pose_retains_complete_marginal_height_certificate()
  {
    auto n=node(.04F);
    n->map_["probability_contract"][0].setConstant(4);
    n->map_.add("probability_pose_event",true,1);
    n->map_.add("height_error_tail_upper",true,.04F);
    n->map_.add("probability_height_radius_cells",false,4);
    n->map_.add("probability_startup_prior_enabled",false,1);
    n->chance_allow_startup_prior_=true;
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    const std::vector<Pose2> path={{0,0,0,0},{1,0,0,1}};
    // Distinct conditional poses alone still exceed the unchanged budget.
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_.add("height_error_radius_m",true,.04F);
    n->map_.add("probability_variance",false,.0001F);
    n->map_.add("probability_height_std_floor",false,0);
    n->map_.add("probability_variance_scale",false,1);
    ASSERT_TRUE(n->probabilityContractValid(n->map_));
    EXPECT_GT(n->queryCostLocked(0,0,0).height_radius,0);
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    // A weak height envelope cannot cancel failed pose-event accounting.
    n->map_["probability_variance"][0].setConstant(.0025F);
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["probability_variance"][0].setConstant(.0001F);
    n->map_["probability_valid"][0].setZero();
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
    n->map_["probability_valid"][0].setOnes();
    n->map_["failure_probability_upper"][0].setConstant(.9F);
    EXPECT_FALSE(n->chancePathSafe(path,nullptr,.1));
  }

  static void forward_progresses_only_on_certified_prefix()
  {
    auto n=node(.02F);n->live_obstacle_enable_=false;
    const Pose2 start{0,0,0,0},goal{3,.4,0,0};
    EXPECT_FALSE(n->chancePathSafe({start,goal},nullptr,.1));
    std::vector<Pose2> prefix;
    ASSERT_TRUE(n->buildCertifiedForwardPrefixLocked(start,goal,prefix));
    EXPECT_GE(prefix.back().s,.25);EXPECT_LT(prefix.back().s,1.2);
    EXPECT_GT(prefix.back().yaw,0);EXPECT_TRUE(n->pathKinematicallyFeasible(prefix));
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=goal;
    EXPECT_TRUE(n->planPath("certified forward fallback"));
    EXPECT_EQ(n->raw_goal_.x,goal.x);EXPECT_EQ(n->raw_goal_.y,goal.y);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    n->map_["probability_valid"][0].setZero();
    EXPECT_FALSE(n->buildCertifiedForwardPrefixLocked(start,goal,prefix));
    n->map_["probability_valid"][0].setOnes();n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->buildCertifiedForwardPrefixLocked(start,goal,prefix));
  }

  static void curved_return_uses_geometry_but_rechecks_support()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    std::vector<Pose2> trail;
    for (int i=0;i<=20;++i) {
      const double x=.1*i;
      trail.push_back(Pose2{x,.15*x*x,std::atan(.3*x),0});
    }
    Pose2 start=trail.back(),goal=trail.front(),guide;
    ASSERT_TRUE(n->reverseGuideTarget(start,goal,trail,guide));
    EXPECT_GT(guide.x,.5);EXPECT_LT(guide.x,1.6);
    std::vector<Pose2> reverse;
    ASSERT_TRUE(n->buildCertifiedReverseArcLocked(start,guide,reverse));
    EXPECT_NEAR(reverse.front().yaw,start.yaw,1e-12);
    EXPECT_EQ(reverse.front().direction,-1);
    EXPECT_TRUE(n->chancePathSafe(reverse,nullptr,.1));
    for (size_t i=1;i<reverse.size();++i) {
      const double ds=reverse[i].s-reverse[i-1].s;
      EXPECT_LE(std::abs(wrapAngle(reverse[i].yaw-reverse[i-1].yaw))/ds,
                n->maxReferenceCurvature()+1e-4);
    }
    EXPECT_FALSE(n->reverseGuideTarget(start,Pose2{0,-1,0,0},trail,guide));
    EXPECT_FALSE(n->reverseGuideTarget(Pose2{2,1.2,start.yaw,0},goal,trail,guide));
    EXPECT_FALSE(n->reverseGuideTarget(Pose2{2,.6,2,0},goal,trail,guide));
    ASSERT_TRUE(n->reverseGuideTarget(start,goal,trail,guide));
    for (auto &slice:n->map_["probability_valid"]) slice.setZero();
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,guide,reverse));
    for (auto &slice:n->map_["probability_valid"]) slice.setOnes();
    n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,guide,reverse));
  }

  static void recorded_corridor_resets_with_odometry()
  {
    auto n=node(1e-6F);
    auto msg=std::make_shared<nav_msgs::msg::Odometry>();
    msg->header.frame_id="map";msg->pose.pose.orientation.w=1;
    msg->header.stamp.sec=10;n->odomCallback(msg);
    msg->header.stamp.nanosec=100000000;msg->pose.pose.position.x=.1;n->odomCallback(msg);
    ASSERT_EQ(n->forward_trail_.size(),2u);
    msg->header.stamp.nanosec=200000000;msg->pose.pose.position.x=0;n->odomCallback(msg);
    EXPECT_EQ(n->forward_trail_.size(),2u);
    msg->header.stamp.sec=9;n->odomCallback(msg);
    EXPECT_EQ(n->forward_trail_.size(),2u); // old delivery cannot rewind geometry
    msg->header.stamp.sec=10;msg->header.stamp.nanosec=300000000;msg->pose.pose.position.x=.1;n->odomCallback(msg);
    ASSERT_EQ(n->forward_trail_.size(),2u); // returning to the last stored point adds no new corridor
    msg->header.frame_id="new_map";msg->header.stamp.nanosec=400000000;n->odomCallback(msg);
    EXPECT_TRUE(n->forward_trail_.empty());EXPECT_TRUE(n->localization_fault_);EXPECT_FALSE(n->odom_.valid);
    msg->header.frame_id="map";
    msg->header.stamp.nanosec=500000000;msg->pose.pose.position.x=.2;n->odomCallback(msg);
    EXPECT_TRUE(n->localization_fault_);EXPECT_TRUE(n->forward_trail_.empty());
  }

  static void reverse_pose_connector_preserves_both_boundary_headings()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    for (double sign : {-1.,1.}) {
      const Pose2 start{1.9,sign*.57,sign*.65,0,-1};
      const Pose2 goal{1.2,sign*.15,sign*.25,0,-1};
      std::vector<Pose2> path;
      ASSERT_TRUE(n->buildCertifiedReversePoseLocked(start,goal,path));
      EXPECT_NEAR(path.front().yaw,start.yaw,1e-12);
      EXPECT_NEAR(path.back().yaw,goal.yaw,1e-5);
      EXPECT_NEAR(path.back().x,goal.x,1e-5);EXPECT_NEAR(path.back().y,goal.y,1e-5);
      for (size_t i=1;i<path.size();++i) {
        const double ds=path[i].s-path[i-1].s;
        EXPECT_EQ(path[i].direction,-1);
        EXPECT_LE(ds,.050001);
        EXPECT_LE(std::abs(wrapAngle(path[i].yaw-path[i-1].yaw))/ds,n->maxReferenceCurvature()+1e-4);
        const double mid=path[i-1].yaw+.5*wrapAngle(path[i].yaw-path[i-1].yaw);
        EXPECT_LT((path[i].x-path[i-1].x)*std::cos(mid)+(path[i].y-path[i-1].y)*std::sin(mid),0);
      }
      EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    }
    const Pose2 start{1.9,.57,.65,0,-1},goal{1.2,.15,.25,0,-1};
    std::vector<Pose2> path;
    for (auto &slice:n->map_["probability_valid"]) slice.setZero();
    EXPECT_FALSE(n->buildCertifiedReversePoseLocked(start,goal,path));
    for (auto &slice:n->map_["probability_valid"]) slice.setOnes();
    n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->buildCertifiedReversePoseLocked(start,goal,path));
  }

  static void short_rear_terminal_connector_is_admissible()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    n->goal_tolerance_=.20;n->goal_reached_distance_=.25;
    n->odom_.valid=true;n->have_goal_=true;n->raw_goal_=Pose2{-.27,0,0,0};
    ASSERT_TRUE(n->planPath("short rear terminal connector"));
    ASSERT_GE(n->path_.size(),2u);
    EXPECT_EQ(n->path_.front().direction,-1);
    EXPECT_NEAR(n->path_.back().x,-.27,1e-6);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
  }

  static void reverse_pose_connector_can_progress_on_a_certified_prefix()
  {
    auto n=node(.02F);n->live_obstacle_enable_=false;
    const Pose2 start{1.9,.57,.65,0,-1},goal{1.2,.15,.25,0,-1};
    std::vector<Pose2> path;
    ASSERT_TRUE(n->buildCertifiedReversePoseLocked(start,goal,path));
    ASSERT_GE(path.size(),3u);
    EXPECT_GE(path.back().s,.25);
    EXPECT_GT(std::hypot(path.back().x-goal.x,path.back().y-goal.y),.01);
    EXPECT_LT(std::abs(wrapAngle(path.back().yaw-goal.yaw)),std::abs(wrapAngle(start.yaw-goal.yaw)));
    EXPECT_TRUE(n->chancePathSafe(path,nullptr,.1));
    EXPECT_TRUE(n->pathKinematicallyFeasible(path));
    for(auto &slice:n->map_["probability_valid"]) slice.setZero();
    EXPECT_FALSE(n->buildCertifiedReversePoseLocked(start,goal,path));
  }

  static void terminal_tolerance_does_not_deadlock_the_last_centimeters()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    n->goal_tolerance_=.22;n->goal_reached_distance_=.25;
    // The true goal is beyond the supported prefix, but its arrival region
    // can be reached without admitting any unsupported state.
    for (int r=0;r<40;++r) for (int c=0;c<40;++c) {
      Eigen::Vector3d position;
      n->map_.index2Pos(Eigen::Array3i(r,c,0),position);
      if(position.x()<-.09) n->map_.at("probability_valid",Eigen::Array3i(r,c,0))=0;
    }
    std::vector<Pose2> prefix;
    ASSERT_TRUE(n->buildCertifiedReverseArcLocked(Pose2{.13,0,0,0,-1},Pose2{-.16,0,0,0,-1},prefix));
    EXPECT_LT(prefix.back().s,.25);
    EXPECT_LE(std::abs(prefix.back().x+.16),n->goal_tolerance_);
    EXPECT_TRUE(n->chancePathSafe(prefix,nullptr,.1));

    n=node(1e-6F);n->live_obstacle_enable_=false;
    n->goal_tolerance_=.22;n->goal_reached_distance_=.25;
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->raw_goal_=Pose2{-.29,0,0,0};
    n->have_path_=true;n->path_={{0,0,0,0,-1},{-.12,0,0,.12,-1}};
    n->controlTimer();
    EXPECT_LT(n->last_cmd_v_,0);
    // The same short prefix toward a distant goal remains an exact stop.
    n->raw_goal_=Pose2{-3,0,0,0};
    n->controlTimer();
    EXPECT_EQ(n->last_cmd_v_,0);
  }

  static void offset_rear_goal_region_still_requires_certified_support()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    n->goal_tolerance_=.22;n->goal_reached_distance_=.25;
    n->odom_.valid=true;n->have_goal_=true;
    // Both the exact-goal curvature and the old rear projection gate fail.
    n->raw_goal_=Pose2{-.23,.13,0,0};
    ASSERT_TRUE(terrain_nav::reverseArc(0,0,0,-.23,.13,4,n->maxReferenceCurvature()).empty());
    ASSERT_TRUE(n->planPath("offset rear goal region"));
    ASSERT_GE(n->path_.size(),2u);
    EXPECT_EQ(n->path_.front().direction,-1);
    EXPECT_LE(std::hypot(n->path_.back().x+.23,n->path_.back().y-.13),.22);
    EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
    EXPECT_TRUE(n->pathKinematicallyFeasible(n->path_));
    const Pose2 start{0,0,0,0,-1};std::vector<Pose2> prefix;
    for (auto &slice:n->map_["probability_valid"]) slice.setZero();
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,n->raw_goal_,prefix));
    for (auto &slice:n->map_["probability_valid"]) slice.setOnes();
    for (auto &slice:n->map_["failure_probability_upper"]) slice.setConstant(.9F);
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,n->raw_goal_,prefix));
    for (auto &slice:n->map_["failure_probability_upper"]) slice.setConstant(1e-6F);
    n->map_["collision_risk"][0].setOnes();
    EXPECT_FALSE(n->buildCertifiedReverseArcLocked(start,n->raw_goal_,prefix));
  }

  static void hybrid_reproduces_bicycle_curvature_in_both_gears()
  {
    auto n=node(1e-6F);n->live_obstacle_enable_=false;
    for (int direction : {-1,1}) for (double curvature : {-.4,.4}) {
      std::vector<Pose2> path;
      for (int i=0;i<=100;++i) {
        const double s=.02*i,theta=direction*s*curvature;
        path.push_back(Pose2{std::sin(theta)/curvature,
          (1-std::cos(theta))/curvature,theta,s,direction});
      }
      OdomState odom;odom.valid=true;odom.v=.3*direction;
      const double correct=std::atan(n->wheel_base_*curvature);
      for (double weight : {0.,.42,1.}) {
        n->pure_pursuit_weight_=weight;
        const auto cmd=n->computePurePursuitCommand(odom,path);
        EXPECT_NEAR(cmd.angular.z,correct,2e-4);
        EXPECT_GT(direction*cmd.linear.x,0);
      }
      n->pure_pursuit_weight_=.42;
      odom.y=.05;
      EXPECT_LT(n->computePurePursuitCommand(odom,path).angular.z,correct);
      odom.y=-.05;
      EXPECT_GT(n->computePurePursuitCommand(odom,path).angular.z,correct);
    }
  }

  static void exhausted_path_stops_exactly()
  {
    auto n=node(.001F);n->live_obstacle_enable_=false;
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    n->have_path_=true;n->path_={{0,0,0,0},{.12,0,0,.12}};
    n->last_cmd_v_=.2;n->last_cmd_steer_=.1;
    n->controlTimer();
    EXPECT_EQ(n->last_cmd_v_,0);EXPECT_EQ(n->last_cmd_steer_,0);
  }

  static void recovery_needs_jointly_certified_forward_exit()
  {
    auto n=node(1e-8F);n->max_steer_=.4;
    const Pose2 start{0,0,0,0},goal{3,0,0,0};
    std::vector<Pose2> retreat,forward_exit;
    ASSERT_TRUE(n->buildRecoveryManeuverLocked(start,goal,retreat,&forward_exit));
    ASSERT_GE(retreat.size(),2u);
    EXPECT_GE(retreat.back().s,.2);EXPECT_LE(retreat.back().s,.8+1e-9);
    EXPECT_LT(retreat.back().x,0);
    EXPECT_TRUE(n->bodyArcPathFeasible(retreat));
    EXPECT_TRUE(n->chancePathSafe(retreat,nullptr,.1));
    for(const auto &p:retreat)EXPECT_EQ(p.direction,-1);
    ASSERT_GE(forward_exit.size(),2u);
    EXPECT_TRUE(n->bodyArcPathFeasible(forward_exit));
    EXPECT_NEAR(forward_exit.front().x,retreat.back().x,1e-9);
    auto whole=retreat;whole.insert(whole.end(),forward_exit.begin(),forward_exit.end());
    n->updatePathArcLength(whole);
    EXPECT_TRUE(n->chancePathSafe(whole,nullptr,.1));
    for(int k=0;k<n->map_.getSizeYaw();++k) {
      auto &valid=n->map_["probability_valid"][k];
      for(int r=0;r<valid.rows();++r)for(int c=0;c<valid.cols();++c) {
        Eigen::Vector3d position;
        n->map_.index2Pos(Eigen::Array3i(r,c,k),position);
        if(position.x()>.05)valid(r,c)=0;
      }
    }
    // The reverse half-plane is safe, but there is no forward exit. It must
    // not be presented as a useful navigation recovery.
    EXPECT_FALSE(n->buildRecoveryManeuverLocked(start,goal,retreat));
    EXPECT_TRUE(retreat.empty());
    EXPECT_FALSE(n->buildRecoveryManeuverLocked(start,Pose2{-3,0,0,0},retreat));
  }

  static void recovery_does_not_reset_risk_or_accept_unknown()
  {
    for(float risk:{.04F,std::numeric_limits<float>::quiet_NaN()}) {
      auto n=node(risk);std::vector<Pose2> retreat;
      EXPECT_FALSE(n->buildRecoveryManeuverLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},retreat));
      EXPECT_TRUE(retreat.empty());
    }
    auto n=node(1e-8F);n->map_["collision_risk"][0].setOnes();
    std::vector<Pose2> retreat;
    EXPECT_FALSE(n->buildRecoveryManeuverLocked(Pose2{0,0,0,0},Pose2{3,0,0,0},retreat));
    EXPECT_TRUE(retreat.empty());
  }

  static void short_reverse_leg_is_not_stopped_before_reorientation()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    n->have_path_=true;n->path_={{0,0,0,0,-1},{-.12,0,0,.12,-1}};
    n->recovery_path_=true;
    n->controlTimer();
    EXPECT_LT(n->last_cmd_v_,0);
    EXPECT_LE(std::abs(n->last_cmd_v_),n->reverse_max_speed_);
    auto mixed=n->path_;mixed.back().direction=1;
    EXPECT_FALSE(n->bodyArcPathFeasible(mixed));
    n->path_commit_enable_=true;
    std::vector<Pose2> previous={{0,0,0,0,-1},{-.2,0,0,.2,-1},{-.4,0,0,.4,-1},{-.6,0,0,.6,-1}};
    std::vector<Pose2> committed;Pose2 search_start;
    EXPECT_FALSE(n->buildCommittedPrefixLocked(n->odom_,previous,committed,search_start));
  }

  static std::unique_ptr<TerrainNavNode> active_recovery()
  {
    auto n=node(1e-8F);n->live_obstacle_enable_=false;
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    n->have_path_=true;n->recovery_path_=true;n->recovery_exit_active_=false;
    n->path_.push_back(Pose2{0,0,0,0,-1});
    for(int i=1;i<=8;++i) {
      auto p=advanceArc(n->path_.back(),.3,-.025);p.s=i*.025;n->path_.push_back(p);
    }
    auto exit=n->path_.back();exit.direction=1;exit.s=0;n->recovery_exit_.push_back(exit);
    for(int i=1;i<=48;++i) {
      auto p=advanceArc(n->recovery_exit_.back(),.3,.025);p.s=i*.025;n->recovery_exit_.push_back(p);
    }
    n->active_goal_=n->path_.back();n->reverse_recovery_count_=1;
    return n;
  }

  static void recovery_keeps_its_exit_and_requires_stationary_cusp()
  {
    auto n=active_recovery();n->odom_.v=-.10;
    const auto original=n->path_.back();
    ASSERT_TRUE(n->planPath("queued ordinary replan"));
    EXPECT_EQ(n->path_.front().direction,-1);EXPECT_FALSE(n->recovery_exit_active_);
    EXPECT_NEAR(n->path_.back().x,original.x,1e-9);
    n->odom_.x=original.x;n->odom_.y=original.y;n->odom_.yaw=original.yaw;
    ASSERT_TRUE(n->planPath("recovery leg complete"));
    EXPECT_EQ(n->path_.front().direction,-1);EXPECT_FALSE(n->recovery_exit_active_);
    n->odom_.v=-.05;
    ASSERT_TRUE(n->planPath("recovery leg complete"));
    EXPECT_EQ(n->path_.front().direction,-1);EXPECT_FALSE(n->recovery_exit_active_);
    n->odom_.v=0;
    ASSERT_TRUE(n->planPath("recovery leg complete"));
    EXPECT_TRUE(n->recovery_path_);EXPECT_TRUE(n->recovery_exit_active_);
    EXPECT_EQ(n->path_.front().direction,1);
    EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
    EXPECT_TRUE(n->pathFootprintSafeLocked(n->path_,.025));
    EXPECT_EQ(n->reverse_recovery_count_,1);
    n->odom_.v=.2;
    EXPECT_FALSE(n->needReplan(n->odom_));
    ASSERT_TRUE(n->planPath("queued ordinary replan"));
    EXPECT_TRUE(n->recovery_exit_active_);EXPECT_EQ(n->path_.front().direction,1);
  }

  static void recovery_rechecks_support_and_cannot_survive_a_new_goal()
  {
    auto n=active_recovery();
    n->odom_.x=n->path_.back().x;n->odom_.y=n->path_.back().y;n->odom_.yaw=n->path_.back().yaw;
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    EXPECT_FALSE(n->planPath("recovery leg complete"));
    EXPECT_FALSE(n->recovery_exit_active_);EXPECT_EQ(n->path_.front().direction,-1);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();
    goal->header.frame_id="map";goal->pose.position.x=2.;goal->pose.position.y=.5;goal->pose.orientation.w=1.;
    n->goalCallback(goal);
    EXPECT_FALSE(n->recovery_path_);EXPECT_TRUE(n->recovery_exit_.empty());
    EXPECT_FALSE(n->have_path_);EXPECT_EQ(n->reverse_recovery_count_,0);
  }

  static void partially_executed_forward_exit_can_release_only_consecutive_retry_guard()
  {
    for(int mode:{0,1,2,3,4}) {
      auto n=active_recovery();n->topology_guide_enable_=true;
      n->path_=n->recovery_exit_;n->recovery_exit_.clear();n->recovery_exit_active_=true;
      const auto &p=n->path_[mode==3?0:mode==4?5:36];n->odom_.x=p.x;n->odom_.y=p.y;n->odom_.yaw=p.yaw;
      if(mode<3)ASSERT_GT(std::hypot(p.x,p.y),.6);
      n->reverse_recovery_count_=2;n->reverse_recovery_total_=7;
      auto origin=std::make_shared<TerrainNavNode::RecoveryProgressOrigin>(
        TerrainNavNode::RecoveryProgressOrigin{mode>=3?-1.:0.,0,n->goal_seq_,n->input_epoch_.load()});
      std::atomic_store(&n->reverse_progress_origin_,std::shared_ptr<const TerrainNavNode::RecoveryProgressOrigin>(origin));
      if(mode==1)for(auto &p:n->path_)p.direction=-1;
      if(mode==2)for(auto &layer:n->map_["probability_valid"])layer.setZero();
      n->planPath("fresh map while forward exit still has remaining distance");
      EXPECT_EQ(n->reverse_recovery_count_,mode==0?0:2);
      EXPECT_EQ(n->reverse_recovery_total_,7);
    }
  }

  static void stopped_recovery_captures_offset_and_recertifies_actual_exit()
  {
    for(int mode:{0,1,2}) {
      auto n=active_recovery();const auto end=n->path_.back();
      n->odom_.x=end.x;n->odom_.y=end.y+(mode==2?.20:.10);n->odom_.yaw=end.yaw;
      n->odom_.v=0.;n->odom_.stamp=n->now();
      if(mode==1) {
        Eigen::Array3i index;
        ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(end.x+.9,end.y+.1,0),index));
        n->map_["collision_risk"][0](index[0],index[1])=1;
      }
      n->controlTimer();
      EXPECT_EQ(n->last_cmd_v_,0.);
      const bool accepted=n->planPath("recovery leg complete");
      if(mode==0) {
        ASSERT_TRUE(accepted);ASSERT_TRUE(n->recovery_exit_active_);
        EXPECT_EQ(n->path_.front().direction,1);
        EXPECT_NEAR(n->path_.front().x,n->odom_.x,1e-9);
        EXPECT_NEAR(n->path_.front().y,n->odom_.y,1e-9);
        EXPECT_TRUE(n->bodyArcPathFeasible(n->path_));
        EXPECT_TRUE(n->pathFootprintSafeLocked(n->path_,.025));
        EXPECT_TRUE(n->chancePathSafe(n->path_,nullptr,.1));
      } else {
        EXPECT_FALSE(accepted);EXPECT_FALSE(n->recovery_exit_active_);
        EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
      }
    }
  }

  static void visualization_uses_navigation_height_before_legacy_estimates()
  {
    auto n=node(1e-8F);
    n->map_.add("probability_elevation",false,-.4F);
    n->map_.add("bgk_mean",false,1.F);n->map_.add("elevation",false,2.F);
    EXPECT_NEAR(n->visualizationTerrainZLocked(0,0,3.),-.4,1e-6);
    n->map_["probability_elevation"][0].setConstant(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(n->visualizationTerrainZLocked(0,0,3.),1.,1e-6);
    n->map_["bgk_mean"][0].setConstant(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(n->visualizationTerrainZLocked(0,0,3.),2.,1e-6);
  }

  static void footprint_recheck_preserves_certified_body_heading()
  {
    auto n=node(1e-8F);
    for(auto &layer:n->map_["probability_valid"])layer.setZero();
    std::vector<Pose2> path={Pose2{.195,0,.17,0}};
    auto end=advanceArc(path.front(),.7,.025);end.s=.025;path.push_back(end);
    for(const auto &p:path) {
      Eigen::Array3i index;
      ASSERT_TRUE(n->map_.pos2Index(Eigen::Vector3d(p.x,p.y,p.yaw),index));
      n->map_.at("probability_valid",index)=1;
    }
    ASSERT_TRUE(n->bodyArcPathFeasible(path));
    EXPECT_TRUE(n->pathFootprintSafeLocked(path,.025));
    auto tangent=path;n->updateYawFromTangent(tangent);
    EXPECT_FALSE(n->chancePathSafe(tangent,nullptr,.1));
  }

  static void control_stops_for_new_obstacle_or_stale_map(bool stale)
  {
    auto n=node(.001F);
    ASSERT_TRUE(n->set_parameter(rclcpp::Parameter("use_sim_time",true)).successful);
    auto *clock=n->get_clock()->get_clock_handle();
    ASSERT_EQ(rcl_enable_ros_time_override(clock),RCL_RET_OK);
    ASSERT_EQ(rcl_set_ros_time_override(clock,10000000000LL),RCL_RET_OK);
    n->odom_.valid=true;n->odom_.stamp=n->now();
    n->have_goal_=true;n->raw_goal_=Pose2{3,0,0,0};
    n->have_path_=true;n->path_={{0,0,0,0},{3,0,0,3}};
    n->last_cmd_v_=.2;n->last_cmd_steer_=.1;
    n->live_obstacle_enable_=true;
    // The new contract repairs a distant invalid suffix. This emergency-stop
    // case must instead put the new obstacle INSIDE the actual stopping
    // reach: 0.8m/s momentum cannot stop in the remaining ~0.35m prefix.
    if(!stale)n->odom_.v=.8;
    auto map=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&n->control_map_snapshot_));
    map->stamp=n->now()-rclcpp::Duration::from_seconds(stale?2.:0.);
    std::atomic_store(&n->control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(map));
    auto live=std::make_shared<LiveObstacleSnapshot>();
    live->stamp_sec=10.;live->min_x=live->min_y=-5.;live->resolution=.1;
    live->width=live->height=100;live->hard.assign(10000,0);
    if (!stale) {
      for (int y=46;y<=54;y++) for(int x=58;x<=62;x++) live->hard[y*100+x]=1;
    }
    std::atomic_store(&n->live_snapshot_,std::shared_ptr<const LiveObstacleSnapshot>(live));
    n->live_dirty_.store(!stale);n->planner_wakeup_=false;
    n->controlTimer();
    EXPECT_EQ(n->last_cmd_v_,0);EXPECT_EQ(n->last_cmd_steer_,0);
    EXPECT_FALSE(n->have_path_);EXPECT_TRUE(n->path_.empty());
    if(stale) EXPECT_TRUE(n->stale_inputs_waiting_.load());
    else {EXPECT_TRUE(n->planner_wakeup_);EXPECT_EQ(n->planner_reason_,"no certified prefix after path invalidation");}
  }

};

class ChanceNavigation : public ::testing::Test
{
protected:
  static void SetUpTestSuite() { if(!rclcpp::ok()) rclcpp::init(0,nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(ChanceNavigation, UnconfirmedMovementStillConstrainsTheCompleteAvoidanceAction) {
  TerrainNavBenchmark::unconfirmed_motion_cannot_disappear_from_a_multisecond_action_proof();
}
TEST_F(ChanceNavigation, ReverseMomentumCannotHideACertifiableForwardEscape) {
  TerrainNavBenchmark::reverse_momentum_does_not_disable_a_certifiable_forward_escape();
}
TEST_F(ChanceNavigation, DynamicHandoverAllowsCertifiedShortReverseTracking) {
  TerrainNavBenchmark::dynamic_handover_checks_the_short_reverse_leg_and_releases_it_to_tracking();
}
TEST_F(ChanceNavigation, DynamicEncounterCanFinishAtANearbyGoalWithoutATerminalPath) {
  TerrainNavBenchmark::nearby_goal_can_be_finished_during_a_dynamic_encounter_without_a_terminal_path();
}
TEST_F(ChanceNavigation, InterruptedRetreatKeepsItsDeadlineAndNeedsAFullFreshProof) {
  TerrainNavBenchmark::active_retreat_can_resume_after_an_interruption_only_with_fresh_proof();
}
TEST_F(ChanceNavigation, CandidateProofRetainsItsUcbWhenExtraSamplingIsDeferred) {
  TerrainNavBenchmark::candidate_proof_can_decline_extra_sampling_without_declining_its_upper_bound();
}
TEST_F(ChanceNavigation, SafeStopDoesNotRemoveAnEarlierCertifiedAccelerationWindow) {
  TerrainNavBenchmark::safe_braking_does_not_remove_an_earlier_certifiable_acceleration_window();
}
TEST_F(ChanceNavigation, AModeledActorCannotLiftAnUnrelatedApproachSpeedCap) {
  TerrainNavBenchmark::safe_braking_does_not_remove_an_earlier_certifiable_acceleration_window(true);
}
TEST_F(ChanceNavigation, RecoverySearchFairnessCannotCrossGoalOrInputEpoch) {
  TerrainNavBenchmark::recovery_search_turn_is_consumed_once_and_scoped_to_goal_and_input();
}
TEST_F(ChanceNavigation, LiveWitnessRefinementDoesNotShrinkBodyOrForecastGeometry) {
  TerrainNavBenchmark::live_witness_geometry_keeps_the_full_body_and_prediction_fallback();
}
TEST_F(ChanceNavigation, LiveWitnessBoundsRetainAllUnexpiredScansAndExpireWithTheirOwnHold) {
  TerrainNavBenchmark::live_witness_history_obeys_the_original_hold_and_does_not_grow_forever();
}
TEST_F(ChanceNavigation, CollisionSubcellsPreserveWitnessesMarginsUnknownAndRollingCoordinates) {
  TerrainNavBenchmark::fine_collision_contract_refines_only_geometry_and_fails_closed_on_bad_data();
}
TEST_F(ChanceNavigation, FinePoseContractRequiresTheCompleteNativeBody) {
  TerrainNavBenchmark::fine_pose_contract_cannot_use_a_smaller_body_or_omit_its_geometry();
}
TEST_F(ChanceNavigation, ContinuousBodyCollisionDoesNotInheritRoundedDisplayPoseVeto) {
  TerrainNavBenchmark::continuous_collision_composes_the_height_bound_without_rounded_pose_veto();
}
TEST_F(ChanceNavigation, ActualCertifiedForwardProgressSurvivesReplacementOfRecoveryPath) {
  TerrainNavBenchmark::ordinary_certified_forward_motion_can_release_an_invalidated_exit_retry_guard();
}
TEST_F(ChanceNavigation, RecoveryProgressCannotComeFromCommandsUnknownReverseOrOldManeuver) {
  TerrainNavBenchmark::recovery_progress_cannot_count_commands_reverse_unknown_or_obsolete_origin();
}
TEST_F(ChanceNavigation, ExhaustedSharedJobCannotAuthorizeUnsearchedMotion) {
  TerrainNavBenchmark::exhausted_shared_job_does_not_publish_path_or_start_recovery();
}

TEST_F(ChanceNavigation, IndexedQueriesMatchNearestWithRollingBuffersAndAllYawSigns) {
  TerrainNavBenchmark::indexed_queries_match_nearest_with_rolling_buffers();
}

TEST_F(ChanceNavigation, EarlyPartialNeedsExistingUsefulFrontierAndFinalCertification) {
  TerrainNavBenchmark::early_partial_requires_useful_existing_frontier_and_explicit_budget();
}
TEST_F(ChanceNavigation, FineLiveGridPreservesExtentBodyChecksAndTerrainGeneration) {
  TerrainNavBenchmark::fine_live_grid_preserves_extent_body_checks_and_terrain_generation();
}

TEST_F(ChanceNavigation, ContractChecksFootprintAndPredictionPermission) { TerrainNavBenchmark::contract(); }
TEST_F(ChanceNavigation, ObservationControllerCapDoesNotLimitCertifiedPaths) {
  TerrainNavBenchmark::observation_controller_caps_only_observation_generation();
}
TEST_F(ChanceNavigation, PendingPublishedInputsCannotDisappearFromCurrentBodyAndStoppingProof) {
  TerrainNavBenchmark::pending_published_input_is_included_in_body_and_stop_admission();
}
TEST_F(ChanceNavigation, CurrentControlRootUsesIssuedAccelerationAndSteeringWithAnUnchangedSourceClock) {
  TerrainNavBenchmark::control_root_replays_actual_interframe_inputs_without_relabeling_source_measurements();
}
TEST_F(ChanceNavigation, DynamicYieldBacksOutOfThreatenedStopWithForwardGoalAndRefusesUnknownRear) {
  TerrainNavBenchmark::dynamic_yield_certifies_time_not_union();
}
TEST_F(ChanceNavigation, FreshCertifiedSafetyRetreatDoesNotWaitForNominalReplanningDebounce) {
  TerrainNavBenchmark::dynamic_yield_certifies_time_not_union(true);
}
TEST_F(ChanceNavigation, TimedYieldKeepsBothLimitsAndExactlyMatchesIndependentProbabilityAccounts) {
  TerrainNavBenchmark::timed_yield_shared_ledger_matches_independent_reference();
}
TEST_F(ChanceNavigation, EarlierStrongerBrakingCannotBeMistakenForSafePassageBeyondACrossing) {
  TerrainNavBenchmark::early_braking_can_stop_in_a_crossing_despite_a_clear_nominal_endpoint();
}
TEST_F(ChanceNavigation, FreshNominalHandoverAvoidsOldSteeringLockAndKeepsSupportOccupancyAndRecoveryGates) {
  TerrainNavBenchmark::fresh_nominal_motion_can_handover_without_hypothetical_old_steering_lock();
}
TEST_F(ChanceNavigation, FreshStoppingSeparationCanEndInteractionDuringOrdinaryPauseButMissingEvidenceCannot) {
  TerrainNavBenchmark::fresh_nominal_motion_can_handover_without_hypothetical_old_steering_lock(true);
}
TEST_F(ChanceNavigation, DynamicConflictEvidenceIsSeparateFromGroundAdmissionAndCannotReleaseFromMissingTargets) {
  TerrainNavBenchmark::geometric_conflict_retirement_never_grants_unknown_ground_or_uses_missing_tracks();
}
TEST_F(ChanceNavigation, BoundedAccelerationReservesOnlyTheCorrespondingTimeAndHeading) {
  TerrainNavBenchmark::dynamic_acceleration_reach_is_directional_and_time_indexed();
}
TEST_F(ChanceNavigation, DynamicManeuverHandoverRequiresAFreshBrakeCertificate) {
  TerrainNavBenchmark::completed_dynamic_maneuver_recertifies_the_handover_stop();
}
TEST_F(ChanceNavigation, DynamicYieldAbortsOnStaleGoalAndClockChanges) {
  TerrainNavBenchmark::dynamic_yield_control_gates_and_cancellation();
}
TEST_F(ChanceNavigation, DynamicCommitRequiresSameFreshFiniteSupportedCertificateAndMission) {
  TerrainNavBenchmark::dynamic_commit_rejects_replaced_expired_or_unsupported_certificates();
}
TEST_F(ChanceNavigation, NormalControlSlowsForMeasuredUnclassifiedBodiesWithoutPaintingRiskOrConfirmingMotion) {
  TerrainNavBenchmark::normal_control_anticipates_compact_cue_before_velocity_confirmation();
}
TEST_F(ChanceNavigation, DynamicAvoidanceAdvancesWhenCertifiedAndCancelsForNewOccupancy) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress();
}
TEST_F(ChanceNavigation, InvalidOldDynamicStopCanSwitchDirectlyToAFreshCertifiedEscape) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(true);
}
TEST_F(ChanceNavigation, OrdinaryRoutePauseCannotBanAnIndependentlyCertifiedForwardEscape) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(false,true);
}
TEST_F(ChanceNavigation, BrakingEnvelopeRetainsMomentumAcrossGearChange) {
  TerrainNavBenchmark::braking_envelope_keeps_momentum_during_gear_change();
}
TEST_F(ChanceNavigation, TimeIndexedCommandShieldChecksAllActorsCurrentSolidsAndUnknownSupport) {
  TerrainNavBenchmark::time_indexed_final_command_rejects_each_actor_and_keeps_current_occupancy();
}
TEST_F(ChanceNavigation, RealCloudMotionForecastIsSeparateFromCurrentRoutingGeometry) {
  TerrainNavBenchmark::measured_cloud_routing_never_turns_future_sweep_into_permanent_geometry();
}
TEST_F(ChanceNavigation, DirectionalBrakeAllowsParallelTravelButRejectsTurnsAndUnknownSupport) {
  TerrainNavBenchmark::directional_braking_distinguishes_parallel_side_walls_from_turning_into_them();
}
TEST_F(ChanceNavigation, DirectionalBrakeRetainsSourceAgeAndConservativeDeceleration) {
  TerrainNavBenchmark::directional_braking_retains_source_latency_and_uses_the_weaker_deceleration();
}
TEST_F(ChanceNavigation, UnknownFrontierIsNotAnInventedSidewaysObstacle) {
  TerrainNavBenchmark::unknown_frontier_does_not_invent_a_sideways_detour();
}
TEST_F(ChanceNavigation, ContinuousArcKeepsAnalyticHeadingAndDirectionalCertificate) {
  TerrainNavBenchmark::continuous_arc_keeps_analytic_heading_and_directional_certificate();
}
TEST_F(ChanceNavigation, ContinuousDubinsPreservesBothBoundaryHeadings) {
  TerrainNavBenchmark::continuous_dubins_preserves_both_boundary_headings();
}
TEST_F(ChanceNavigation, ContinuousFrenetCertifiesAnalyticBoundaryHeadings) {
  TerrainNavBenchmark::continuous_frenet_certifies_analytic_boundary_headings();
}
TEST_F(ChanceNavigation, ContinuousReplanKeepsCurvedCommittedPrefix) {
  TerrainNavBenchmark::continuous_replan_keeps_curved_committed_prefix();
}
TEST_F(ChanceNavigation, CertifiedTurnIsNotStarvedByStraightObservationPrefix) {
  TerrainNavBenchmark::certified_turn_precedes_observation_only_progress();
}
TEST_F(ChanceNavigation, ObservationFallbackKeepsCommittedConnectorAndJointCertificate) {
  TerrainNavBenchmark::observation_fallback_keeps_committed_connector();
}
TEST_F(ChanceNavigation, GuideContinuityYieldsToNewCollisionEvidence) {
  TerrainNavBenchmark::guide_continuity_is_soft_and_never_admits_blocked_side();
}
TEST_F(ChanceNavigation, InflatedLiveCellsDoNotDoubleInflateBodyButRealCollisionsBlock) {
  TerrainNavBenchmark::inflated_live_body_is_not_a_second_tracking_envelope();
}
TEST_F(ChanceNavigation, BodyCellOverlapChecksAllFourAxes) {
  TerrainNavBenchmark::grid_cell_overlap_uses_all_separating_axes();
}
TEST_F(ChanceNavigation, StateSafeDoesNotMeanPathWithinBudget) { TerrainNavBenchmark::joint_budget(); }
TEST_F(ChanceNavigation, UnsupportedDoesNotFallBackToBubble) { TerrainNavBenchmark::unknown_cannot_use_legacy_bubble(); }
TEST_F(ChanceNavigation, NarrowCollisionStillBlocks) { TerrainNavBenchmark::one_cell_collision_is_not_a_probability_average(); }
TEST_F(ChanceNavigation, GuideSearchDoesNotHoldControlGoalLock) {
  TerrainNavBenchmark::guide_search_uses_snapshot_without_holding_control_goal_lock();
}
TEST_F(ChanceNavigation, FinalGoalStaysFixedWhenRollingSubgoalChanges) {
  TerrainNavBenchmark::final_goal_marker_is_not_replaced_by_rolling_subgoal();
}
TEST_F(ChanceNavigation, OptimisticTopologyCannotBypassUnknownTerrain) {
  TerrainNavBenchmark::optimistic_topology_guide_cannot_authorize_unknown_execution();
}
TEST_F(ChanceNavigation, GlobalGuideExecutesOnlyContinuousFullyCertifiedPrefix) {
  TerrainNavBenchmark::topology_prefix_stops_at_unknown_and_checks_full_body();
}
TEST_F(ChanceNavigation, RecoveryRequiresJointCertifiedRetreatAndGlobalRouteExit) {
  TerrainNavBenchmark::topology_recovery_certifies_both_legs_and_unknown_never_passes();
}
TEST_F(ChanceNavigation, GlobalRoutingReusesComfortWithoutExtraInflation) {
  TerrainNavBenchmark::global_routing_reuses_soft_clearance_without_changing_probability_or_collision();
}
TEST_F(ChanceNavigation, RetainedPathStillNeedsCurrentBodySupportAndJointBudget) {
  TerrainNavBenchmark::path_retention_requires_current_entire_body_and_joint_risk_not_old_certificate();
}
TEST_F(ChanceNavigation, LongerFreshGuideDoesNotFlipACertifiedNearApproach) {
  TerrainNavBenchmark::a_longer_new_guide_cannot_flip_a_still_certified_near_approach();
}
TEST_F(ChanceNavigation, TangentBeyondEndpointTriggersSearchWithoutRevokingFiniteCurve) {
  TerrainNavBenchmark::tangent_obstacle_is_a_search_hint_not_a_finite_curve_collision();
}
TEST_F(ChanceNavigation, NearbySideObstacleAndFailedSearchDoNotOverrideCertifiedForwardMotion) {
  TerrainNavBenchmark::search_failure_near_side_obstacle_cannot_reverse_a_certified_forward_path();
}
TEST_F(ChanceNavigation, CompletedReverseGuideIsHandedToStoppedExecutorInsteadOfWaitingAgain) {
  TerrainNavBenchmark::completed_reverse_guide_is_dispatched_by_the_actual_planner_job();
}
TEST_F(ChanceNavigation, CompleteForwardGeometryCannotStarveASupportedReverseEntrance) {
  TerrainNavBenchmark::complete_forward_geometry_without_supported_entrance_cannot_starve_reverse_search();
}
TEST_F(ChanceNavigation, FarMapOrLiveCollisionRetainsOnlyARecertifiedFinitePrefix) {
  TerrainNavBenchmark::far_path_collision_keeps_only_the_current_certified_prefix();
}
TEST_F(ChanceNavigation, PrefixRepairStopsAtUnknownAndCannotMaskCurrentBodyCollision) {
  TerrainNavBenchmark::prefix_repair_never_bridges_unknown_or_a_current_body_collision();
}
TEST_F(ChanceNavigation, PrefixRepairReservesStoppingDistanceAndChecksPathIdentity) {
  TerrainNavBenchmark::repaired_prefix_requires_stop_distance_and_cannot_overwrite_a_new_path();
}
TEST_F(ChanceNavigation, TopologyContinuationIsSoftAndCannotAuthorizeOldGroundOrGeometry) {
  TerrainNavBenchmark::topology_continuity_is_a_finite_preference_and_never_a_certificate();
}
TEST_F(ChanceNavigation, EarlyContinuationSearchDoesNotReplaceGoodCertificateWithShorterFrontier) {
  TerrainNavBenchmark::early_continuation_search_cannot_discard_a_better_current_certificate();
}
TEST_F(ChanceNavigation, RecoveryProgressUsesRawSourceOdometryAcrossPredictedControlTicks) {
  TerrainNavBenchmark::control_pose_prediction_must_not_replace_source_recovery_progress();
}
TEST_F(ChanceNavigation, FiveCentimeterObstacleGeometryRetainsFullCarAndUnknownRefusal) {
  TerrainNavBenchmark::finer_obstacle_geometry_does_not_shrink_the_protected_car_or_padding();
}
TEST_F(ChanceNavigation, HighUpperBoundDoesNotBecomeAFictitiousGlobalWall) {
  TerrainNavBenchmark::high_probability_upper_bound_is_soft_routing_but_hard_execution();
}
TEST_F(ChanceNavigation, RecoveryBudgetResetsOnlyAfterActualForwardExitProgress) {
  TerrainNavBenchmark::only_completed_displaced_forward_exit_resets_consecutive_recovery_limit();
}
TEST_F(ChanceNavigation, RecoveryCaptureRechecksActualOffsetAndRejectsBlockedExit) {
  TerrainNavBenchmark::stopped_recovery_captures_offset_and_recertifies_actual_exit();
}
TEST_F(ChanceNavigation, FailedLocalGuideCannotVetoCertifiedMissionTurningProgress) {
  TerrainNavBenchmark::mission_progress_fan_turns_after_failed_guide_without_relaxing_safety();
}
TEST_F(ChanceNavigation, PartialForwardExitProgressReleasesOnlyConsecutiveRetryGuard) {
  TerrainNavBenchmark::partially_executed_forward_exit_can_release_only_consecutive_retry_guard();
}
TEST_F(ChanceNavigation, VisualizationUsesNavigationElevation) {
  TerrainNavBenchmark::visualization_uses_navigation_height_before_legacy_estimates();
}
TEST_F(ChanceNavigation, LiveUpperReturnsUseOnlyFreshMeasuredGround) { TerrainNavBenchmark::live_columns_require_measured_fresh_ground(); }
TEST_F(ChanceNavigation, RaisedBumperUsesBodyEnvelopeNotGroundStencil) { TerrainNavBenchmark::raised_bumper_does_not_shrink_to_ground_contact_stencil(); }
TEST_F(ChanceNavigation, SampledControlFailsClosed) { TerrainNavBenchmark::sampled_control_fails_closed(); }
TEST_F(ChanceNavigation, ValidSampledSpeedIsNotRaised) { TerrainNavBenchmark::sampled_control_does_not_raise_a_safe_speed(); }
TEST_F(ChanceNavigation, DoesNotAdvertiseMppi) { TerrainNavBenchmark::controller_names_are_not_mppi(); }
TEST_F(ChanceNavigation, GoalRunsCompletePlanner) { TerrainNavBenchmark::goal_runs_complete_planner(); }
TEST_F(ChanceNavigation, ShortTerminalConnectorIsNotRejectedAsUnproductiveFrontier) { TerrainNavBenchmark::short_terminal_connector_is_not_a_failed_frontier(); }
TEST_F(ChanceNavigation, OffAxisObservationPrefixRetainsTurningRoom) { TerrainNavBenchmark::off_axis_observation_retains_turning_room(); }
TEST_F(ChanceNavigation, ReversePoseConnectorKeepsBothBoundaryHeadings) { TerrainNavBenchmark::reverse_pose_connector_preserves_both_boundary_headings(); }
TEST_F(ChanceNavigation, ReversePoseConnectorRetainsHeadingOnPartialCertificate) { TerrainNavBenchmark::reverse_pose_connector_can_progress_on_a_certified_prefix(); }
TEST_F(ChanceNavigation, ShortRearTerminalConnectorIsAdmissible) { TerrainNavBenchmark::short_rear_terminal_connector_is_admissible(); }
TEST_F(ChanceNavigation, TerminalToleranceDoesNotDeadlockLastCentimeters) { TerrainNavBenchmark::terminal_tolerance_does_not_deadlock_the_last_centimeters(); }
TEST_F(ChanceNavigation, OffsetRearGoalRegionStillRequiresCertifiedSupport) { TerrainNavBenchmark::offset_rear_goal_region_still_requires_certified_support(); }
TEST_F(ChanceNavigation, PlanarLayersIgnoreHeading) { TerrainNavBenchmark::planar_layers_ignore_heading(); }
TEST_F(ChanceNavigation, UnknownGoalRejectsBeforeSearch) { TerrainNavBenchmark::unknown_goal_rejects_before_search(); }
TEST_F(ChanceNavigation, SharedSourceBudgetReusesGround) { TerrainNavBenchmark::shared_source_budget_reuses_ground_not_pose_events(); }
TEST_F(ChanceNavigation, StartupAssumptionRequiresSimulationPermission) { TerrainNavBenchmark::startup_assumption_requires_explicit_simulation_permission(); }
TEST_F(ChanceNavigation, SharedGpRequiresPermissionAndKeepsBudgetUnknownAndCollisionChecks) { TerrainNavBenchmark::shared_gp_requires_permission_and_preserves_navigation_checks(); }
TEST_F(ChanceNavigation, ReverseCanUseCertifiedPrefixWithoutLooseningBudget) { TerrainNavBenchmark::reverse_progresses_only_on_certified_prefix(); }
TEST_F(ChanceNavigation, OffsetReverseUsesSafeGoalDisk) { TerrainNavBenchmark::offset_reverse_goal_uses_certified_terminal_disk(); }
TEST_F(ChanceNavigation, KnownHazardPreemptsStraightObservationPrefix) { TerrainNavBenchmark::observation_prefix_does_not_override_known_hazard(); }
TEST_F(ChanceNavigation, CertifiedShortLatticeFrontierIsNotDiscarded) { TerrainNavBenchmark::certified_short_lattice_frontier_is_useful_progress(); }
TEST_F(ChanceNavigation, LatticeKeepsContinuousArcAndCertifiedBodyHeading) { TerrainNavBenchmark::lattice_keeps_continuous_arc_and_certified_body_heading(); }
TEST_F(ChanceNavigation, TerminalReverseCertifiesAlternativeArcs) { TerrainNavBenchmark::terminal_reverse_checks_alternative_arcs_before_stopping(); }
TEST_F(ChanceNavigation, NearbyRearGoalCanUseCertifiedParkingManeuverButCannotSkipStoppedCuspOrGroundRiskProof) {
  TerrainNavBenchmark::terminal_two_gear_connector_preserves_support_risk_and_stopped_handover();
}
TEST_F(ChanceNavigation, ConditionalPoseRetainsMarginalSourceCertificate) { TerrainNavBenchmark::conditional_pose_retains_complete_marginal_height_certificate(); }
TEST_F(ChanceNavigation, ForwardCanUseCertifiedPrefixWithoutLooseningBudget) { TerrainNavBenchmark::forward_progresses_only_on_certified_prefix(); }
TEST_F(ChanceNavigation, CurvedReturnKeepsCurrentSupportAndCollisionChecks) { TerrainNavBenchmark::curved_return_uses_geometry_but_rechecks_support(); }
TEST_F(ChanceNavigation, ReturnCorridorDoesNotSurviveOdomReset) { TerrainNavBenchmark::recorded_corridor_resets_with_odometry(); }
TEST_F(ChanceNavigation, HybridPreservesCircleSteeringInBothGears) { TerrainNavBenchmark::hybrid_reproduces_bicycle_curvature_in_both_gears(); }
TEST_F(ChanceNavigation, ExhaustedPathStopsExactly) { TerrainNavBenchmark::exhausted_path_stops_exactly(); }
TEST_F(ChanceNavigation, RecoveryNeedsCertifiedForwardExit) { TerrainNavBenchmark::recovery_needs_jointly_certified_forward_exit(); }
TEST_F(ChanceNavigation, DeadendRetreatRequiresCurrentWholeBodyGroundAndJointRisk) {
  TerrainNavBenchmark::deadend_retreat_is_finite_and_requires_current_whole_body_ground_and_budget();
}
TEST_F(ChanceNavigation, DeadendRetreatStopsAndReplansWithoutInventingAForwardExit) {
  TerrainNavBenchmark::deadend_retreat_stops_before_replanning_without_claiming_a_forward_exit();
}
TEST_F(ChanceNavigation, SuspendedRecoveryNeedsFreshFullRevalidationAndKeepsItsAttemptCount) {
  TerrainNavBenchmark::paused_recovery_requires_fresh_recertification_and_does_not_consume_another_attempt();
}
TEST_F(ChanceNavigation, TemporalWaitDoesNotEraseCertifiedForwardProgressOrWaiveFrameIdentity) {
  TerrainNavBenchmark::temporal_pauses_preserve_measured_forward_progress_but_frame_changes_do_not_release_retries();
}
TEST_F(ChanceNavigation, SpatialRoutingMemorySurvivesTemporalPauseAndRejectsAChangedFrame) {
  TerrainNavBenchmark::routing_geometry_survives_temporal_pause_but_not_a_spatial_generation_change();
}
TEST_F(ChanceNavigation, FailedBranchPreferenceSurvivesTheReverseUntilTheActualRouteDecision) {
  TerrainNavBenchmark::failed_branch_preference_is_refreshed_at_the_stopped_cusp_without_changing_risk();
}
TEST_F(ChanceNavigation, GlobalSelectionDoesNotPreferAFasterLocalEntranceIntoADeadEnd) {
  TerrainNavBenchmark::global_proposal_selection_uses_the_full_frontier_after_certifying_its_local_prefix();
}
TEST_F(ChanceNavigation, StagedGearPlanStopsAndRevalidatesAtEachCusp) {
  TerrainNavBenchmark::staged_gears_stop_and_recertify_every_direction_change();
}
TEST_F(ChanceNavigation, StagedGearPlanRejectsChangedSupportGeometryAgeFrameAndTransitionBudget) {
  TerrainNavBenchmark::staged_gears_cannot_dispatch_unknown_stale_colliding_or_over_budget_legs();
}
TEST_F(ChanceNavigation, StagedReverseRetainsBodyHeadingAndItsNextForwardCusp) {
  TerrainNavBenchmark::staged_local_join_preserves_world_fixed_future_gates_and_goal();
}
TEST_F(ChanceNavigation, RecoveryCannotResetBudgetOrAcceptUnknown) { TerrainNavBenchmark::recovery_does_not_reset_risk_or_accept_unknown(); }
TEST_F(ChanceNavigation, ShortReverseLegCanReachReorientationPose) { TerrainNavBenchmark::short_reverse_leg_is_not_stopped_before_reorientation(); }
TEST_F(ChanceNavigation, RecoveryTracksBothLegsAndStopsAtCusp) { TerrainNavBenchmark::recovery_keeps_its_exit_and_requires_stationary_cusp(); }
TEST_F(ChanceNavigation, RecoveryRechecksSupportAndNewGoalCancelsIt) { TerrainNavBenchmark::recovery_rechecks_support_and_cannot_survive_a_new_goal(); }
TEST_F(ChanceNavigation, FootprintRecheckPreservesCertifiedHeading) { TerrainNavBenchmark::footprint_recheck_preserves_certified_body_heading(); }
TEST_F(ChanceNavigation, NewLiveObstacleStopsProbabilityController) { TerrainNavBenchmark::control_stops_for_new_obstacle_or_stale_map(false); }
TEST_F(ChanceNavigation, StaleMapStopsProbabilityController) { TerrainNavBenchmark::control_stops_for_new_obstacle_or_stale_map(true); }
TEST_F(ChanceNavigation, ReserveRecoveryRetainsPhysicalMarginAndRequiresReleasedExit) {
  TerrainNavBenchmark::reserve_recovery_keeps_physical_margin_and_exits_reserve();
}
TEST_F(ChanceNavigation, ReserveRecoveryCannotBypassUnknownStaleProbabilityOrMovingObstacle) {
  TerrainNavBenchmark::reserve_recovery_rejects_unknown_stale_probability_and_crossing();
}
TEST_F(ChanceNavigation, ReserveRecoveryIsStationaryBoundedAndGoalCancellable) {
  TerrainNavBenchmark::reserve_recovery_control_is_bounded_and_cancelled();
}
TEST_F(ChanceNavigation, ReserveAndManeuverAttemptsAreSeparateAndDuplicateGoalCannotResetThem) {
  TerrainNavBenchmark::reserve_and_maneuver_attempts_are_independent_and_reset_for_new_goal();
}
TEST_F(ChanceNavigation, SideObstacleRecoveryStillRequiresKnownCertifiedRetreatAndExit) {
  TerrainNavBenchmark::side_obstacle_can_trigger_certified_recovery_but_not_unknown_motion();
}
TEST_F(ChanceNavigation, ReserveRecoveryConfirmsActualStopAndNewMapBeforeCompletion) {
  TerrainNavBenchmark::reserve_recovery_completes_only_after_stopped_fresh_release();
}
TEST_F(ChanceNavigation, SameXyGroundCannotPrematurelyClearThinUpperObstacle) {
  TerrainNavBenchmark::same_xy_ground_does_not_erase_a_thin_upper_obstacle();
}
TEST_F(ChanceNavigation, RecoveryRouteMemoryIsSoftBoundedAndGoalScoped) {
  TerrainNavBenchmark::recovery_route_memory_is_soft_bounded_and_goal_scoped();
}
TEST_F(ChanceNavigation, RecoveryRouteMemoryExpiresAndCannotReleaseUnsafeSupport) {
  TerrainNavBenchmark::recovery_route_memory_expires_and_never_releases_unsafe_support();
}
TEST_F(ChanceNavigation, PrimitiveClearanceMatchesFinalPathAndControlGuard) {
  TerrainNavBenchmark::primitive_clearance_matches_final_path_and_control_guard();
}
TEST_F(ChanceNavigation, ReserveExitNeedsForwardProgressAndJointSourceEventBudget) {
  TerrainNavBenchmark::reserve_forward_exit_requires_progress_and_shared_budget();
}
TEST_F(ChanceNavigation, ReserveExitHandoffRevalidatesActualPoseAndRejectsObsoletePlanner) {
  TerrainNavBenchmark::reserve_exit_handoff_recertifies_actual_pose_and_blocks_old_planner();
}
TEST_F(ChanceNavigation, HistoricalObstacleAdvisoryCannotChangeSupportProbabilityOrControlClearance) {
  TerrainNavBenchmark::historical_obstacles_are_planning_preferences_not_safety_events();
}
TEST_F(ChanceNavigation, CollisionEnvelopeUsesVehicleWidthHeadingAndOneMargin) {
  TerrainNavBenchmark::collision_envelope_uses_width_heading_and_exactly_one_margin();
}
TEST_F(ChanceNavigation, HeightSliceNeedsMatchingDimensionsFreshMapAndKeepsLowBar) {
  TerrainNavBenchmark::live_height_slice_requires_matching_geometry_and_fresh_map();
}
TEST_F(ChanceNavigation, FreshHeightProofClearsOnlyHeldAllOverheadGeometry) {
  TerrainNavBenchmark::held_overhead_cannot_outlive_fresh_height_evidence_but_low_obstacles_do();
}
TEST_F(ChanceNavigation, MotionAggregationPreservesFineCollisionGeometry) {
  TerrainNavBenchmark::motion_grid_aggregation_preserves_collision_resolution_and_extent();
}
TEST_F(ChanceNavigation, ProbabilityLiveGeometryKeepsTwentyCentimeterStepBlocked) {
  TerrainNavBenchmark::probability_live_columns_reject_low_step_above_physical_limit();
}
TEST_F(ChanceNavigation, ProvisionalMotionCanBrakeButCannotReverseOrChangeTerrainEvidence) {
  TerrainNavBenchmark::provisional_motion_can_only_brake_without_changing_path_or_support();
}
TEST_F(ChanceNavigation, DerivedCloudReliabilityIsExplicitAndKeepsLatest) {
  TerrainNavBenchmark::reliable_registered_cloud_keeps_latest_without_changing_generic_sensor_policy();
}
TEST_F(ChanceNavigation, FirstTranslationVelocityReserveOnlyGrowsBrakePrediction) {
  TerrainNavBenchmark::uncertain_first_translation_expands_only_the_braking_prediction();
}
TEST_F(ChanceNavigation, IngressUpdatesControlAndInvalidatesContractWithoutSearchMutex) {
  TerrainNavBenchmark::map_ingress_cannot_wait_for_search_and_invalidates_immediately();
}
TEST_F(ChanceNavigation, FirstGoalAckSurvivesZeroClockAndLateDiscovery) {
  TerrainNavBenchmark::first_goal_ack_survives_zero_clock_and_late_discovery();
}
TEST_F(ChanceNavigation, StuckStopKeepsStatusCurrentAndRequiresNewGoal) {
  TerrainNavBenchmark::stuck_stop_keeps_status_current_and_requires_a_new_goal();
}
TEST_F(ChanceNavigation,TractionRecoveryRequiresStoppedFreshPlanningAndRevokesPrestopPaths) {
  TerrainNavBenchmark::traction_replan_revokes_stalled_and_prestop_worker_paths();
}

TEST_F(ChanceNavigation, LiveRoutingClearanceRequiresDistinctActualGroundObservations) {
  TerrainNavBenchmark::live_routing_clearance_uses_actual_distinct_scans_and_preserves_occlusion();
}

TEST_F(ChanceNavigation, HistoricalRoutingGeometrySurvivesFloorOnlyReturnsAndRequiresFreshMapperToClear) {
  TerrainNavBenchmark::routing_history_preserves_a_wall_when_current_floor_returns_do_not_see_it();
}

TEST_F(ChanceNavigation, CertifiedShortTopologyPrefixCanObserveWithoutExecutingUnknownAhead) {
  TerrainNavBenchmark::supported_short_topology_prefix_is_not_discarded_by_a_long_endpoint_threshold();
}

TEST_F(ChanceNavigation, AdvisoryStartOverlapRequiresFreshIndependentActualEvidence) {
  TerrainNavBenchmark::historical_start_relief_needs_a_current_independent_state_certificate();
}

TEST_F(ChanceNavigation, InvalidOdometryLatchesStopAndCannotBeHiddenBySpeedClampingOrANewPlan) {
  TerrainNavBenchmark::impossible_odometry_cannot_be_clamped_into_a_valid_control_state();
}

TEST_F(ChanceNavigation, FutureGearChangesInformRoutingButNeverBecomeUncertifiedForwardExecution) {
  TerrainNavBenchmark::global_future_reverse_is_never_smuggled_into_a_forward_guide();
}

TEST_F(ChanceNavigation, HistoricalExtentKeepsConservativeFallbackWithoutValidFineWitnesses) {
  TerrainNavBenchmark::historical_subcells_preserve_whole_square_fallback();
}

TEST_F(ChanceNavigation, MeasuredFineGapPassesOneProtectedBodyAndStillRejectsImpossibleTurn) {
  TerrainNavBenchmark::one_protected_body_passes_a_real_gap_but_cannot_turn_through_its_walls();
}

TEST_F(ChanceNavigation, UnknownExplorationCostsMoreThanCertifiedTerrainWithoutBecomingExecutable) {
  TerrainNavBenchmark::optimistic_exploration_never_prices_unknown_below_a_certified_pose();
}

TEST_F(ChanceNavigation, LocalRecoveryCertifiesBothLegsWithoutRequiringTheDistantFinalGoalToBeReached) {
  TerrainNavBenchmark::local_recovery_can_use_a_useful_frontier_without_claiming_final_arrival();
}

TEST_F(ChanceNavigation, HistoricalSoftCostCannotActAsAnotherHardCorridorInflation) {
  TerrainNavBenchmark::a_soft_historical_preference_cannot_close_a_physically_open_corridor();
}
TEST_F(ChanceNavigation, RevalidatedUsefulForwardPathPrecedesFreshOptimisticTopology) {
  TerrainNavBenchmark::certified_forward_path_is_revalidated_before_a_different_topology_is_searched();
}

TEST_F(ChanceNavigation, ControlPathValidationDoesNotAddAnotherInflationToACertifiedNarrowGap) {
  TerrainNavBenchmark::control_and_search_use_the_same_protected_gap_and_probability_budget();
}

TEST_F(ChanceNavigation, HunterRearAxleReferenceMatchesHybridTrackingInBothGears) {
  TerrainNavBenchmark::offset_body_and_rear_tracking_agree_in_both_gears();
}
TEST_F(ChanceNavigation, VaryingCurvatureFeedforwardIsNotAttenuatedByPreviewBlending) {
  TerrainNavBenchmark::varying_curvature_reference_has_correct_feedforward_in_each_gear();
}
TEST_F(ChanceNavigation, ClearanceSpanContractCannotRaiseStepLimitOrInventUnderbodyClearance) {
  TerrainNavBenchmark::clearance_span_contract_is_separate_from_local_step_and_fails_closed();
}
TEST_F(ChanceNavigation, HunterOffsetArcAndFrenetConnectorsKeepActualBaseEndpoints) {
  TerrainNavBenchmark::offset_connectors_keep_base_endpoints_and_full_body_heading();
}
TEST_F(ChanceNavigation, HunterOffsetStagedRecoveryCanStopReverseAndResumeForward) {
  TerrainNavBenchmark::offset_staged_plan_recertifies_reverse_then_forward();
}

TEST_F(ChanceNavigation, StagedRecoveryExecutesOnlyTheCertifiedPrefixAndKeepsItsUnknownSameGearTail) {
  TerrainNavBenchmark::staged_execution_keeps_uncertified_future_tail_without_executing_it();
}

TEST_F(ChanceNavigation, UnknownReverseRoutePreferenceCannotCertifyUnknownGround) {
  TerrainNavBenchmark::unsupported_reverse_cost_is_finite_and_never_changes_admission();
}
TEST_F(ChanceNavigation, FinitePrefixCertificateWorksInBothGearsAndRejectsUnstoppedCusp) {
  TerrainNavBenchmark::finite_prefix_certificate_handles_each_gear_but_never_a_hidden_cusp();
}
TEST_F(ChanceNavigation, ObsoleteGuideHandoffCannotDeleteNewerPrefixCertificate) {
  TerrainNavBenchmark::obsolete_guide_wait_cannot_erase_a_newly_certified_path();
}
TEST_F(ChanceNavigation, CompletedReverseRouteDispatchesEvenWhenStoppingPrefixRetentionSucceeded) {
  TerrainNavBenchmark::ready_reverse_cannot_be_starved_by_a_successful_stopping_prefix();
}
TEST_F(ChanceNavigation, RecoverySuffixRepairRetainsSameGearTailBeforeWorldFixedNextLeg) {
  TerrainNavBenchmark::repaired_recovery_preserves_same_gear_tail_and_world_fixed_next_leg();
}
TEST_F(ChanceNavigation, ShortCertifiedCuspRemainderCanUseTightlyCapturedWorldFixedReference) {
  TerrainNavBenchmark::short_certified_cusp_remainder_is_not_an_unknown_observation_prefix();
}

TEST_F(ChanceNavigation, CompleteForwardSeedCannotBypassUnknownSupportOrKnownUnsafeTerrain) {
  TerrainNavBenchmark::complete_forward_seed_requires_local_certificate_but_future_is_only_guidance();
}
TEST_F(ChanceNavigation, RetainedForwardReferenceCertifiesOnlySupportedPrefixAndMatchesSpaceGoalCapture) {
  TerrainNavBenchmark::retained_forward_reference_recertifies_after_time_epoch_but_never_space_or_hazard();
}
TEST_F(ChanceNavigation, CompleteRouteGuidanceDoesNotRequireExecutingUnknownSuffix) {
  TerrainNavBenchmark::complete_seed_can_guide_beyond_a_short_certified_observation_prefix();
}
TEST_F(ChanceNavigation, BackgroundRouteCannotBypassChangedFrameGoalAgeOrCurrentSupport) {
  TerrainNavBenchmark::background_results_are_only_guidance_and_require_current_identity_and_support();
}
TEST_F(ChanceNavigation, BackgroundSearchOwnsItsInputAndCannotExecuteAfterSupportDisappears) {
  TerrainNavBenchmark::background_search_uses_immutable_copies_but_adoption_uses_new_support();
}
TEST_F(ChanceNavigation, BackgroundProposalDiversityUsesOneCommonRiskObjective) {
  TerrainNavBenchmark::background_candidate_ranking_keeps_risk_even_for_geometric_proposal();
}
TEST_F(ChanceNavigation, LooseMeanSafeRiskBoundCannotMakeObservedGroundWorseThanUnknownGuidance) {
  TerrainNavBenchmark::mean_safe_uncertainty_is_not_costlier_guidance_than_wholly_unknown_ground();
}
TEST_F(ChanceNavigation, TrackingReserveIsAFinitePreferenceAndDoesNotChangeSafetyEnvelope) {
  TerrainNavBenchmark::tracking_reserve_prefers_clearance_without_turning_it_into_a_hard_margin();
}
TEST_F(ChanceNavigation, NarrowPassagesRetainNominalSearchWhenPreferredTrackingReserveCannotFit) {
  TerrainNavBenchmark::unavailable_preferred_reserve_does_not_block_the_nominal_narrow_route();
}
TEST_F(ChanceNavigation, ObservationProposalDoesNotClearHistoryOrExecuteWithoutCurrentCertificate) {
  TerrainNavBenchmark::observation_background_never_changes_history_or_authorizes_unknown_execution();
}
TEST_F(ChanceNavigation, CompleteManeuverRetainsFutureGearsWithoutExecutingUnknownOrMovingGearChanges) {
  TerrainNavBenchmark::background_maneuver_keeps_future_gears_but_executes_only_a_fresh_stopped_prefix();
}
TEST_F(ChanceNavigation, BackgroundSearchFindsReverseThenForwardRouteBeyondCurrentBlocker) {
  TerrainNavBenchmark::background_search_can_find_a_complete_reverse_first_maneuver();
}
TEST_F(ChanceNavigation, GlobalPlanningPauseCannotWaitForeverAtTheSameBranch) {
  TerrainNavBenchmark::whole_goal_search_pause_is_bounded_and_scoped_to_progress_and_goal();
}
TEST_F(ChanceNavigation, FarUnknownTailKeepsCurrentlyCertifiedTurningPrefixWithoutReusingOldCertificate) {
  TerrainNavBenchmark::far_invalid_suffix_does_not_erase_the_current_certified_turn();
}
TEST_F(ChanceNavigation, WitnessRectanglesRetainSensorPaddingCarMarginAndUnknownRejection) {
  TerrainNavBenchmark::measured_subcell_boxes_do_not_add_unwitnessed_square_extent();
}
TEST_F(ChanceNavigation, BoundedClearancePreservesExactClippedEightConnectedDistances) {
  TerrainNavBenchmark::bounded_clearance_matches_unbounded_octile_distances();
}

TEST_F(ChanceNavigation, BoundedMeanContractRequiresExplicitPermissionAndCannotUseGaussianMarginalShortcut) {
  TerrainNavBenchmark::bounded_mean_contract_requires_permission_and_only_reuses_pose_events();
}
TEST_F(ChanceNavigation, InterruptedRecoveryKeepsIntentButMustStopAndRecertifyBeforeNextGear) {
  TerrainNavBenchmark::recovery_evidence_wait_preserves_intent_without_permission_or_extra_attempts();
}
TEST_F(ChanceNavigation, RecoveryEvidenceWaitCannotRenewItsDeadlineOrCrossGoalEpochPathIdentity) {
  TerrainNavBenchmark::recovery_wait_deadline_is_not_renewed_and_cannot_cross_identity();
}
TEST_F(ChanceNavigation, LostConflictPredictionKeepsStoppingAcrossTicksUntilPositiveResolution) {
  TerrainNavBenchmark::lost_dynamic_prediction_does_not_resume_on_the_next_tick();
}
TEST_F(ChanceNavigation, FiniteForwardEscapeCanStopBeforeUnsupportedFrontierInsteadOfMissingBothLongEndpoints) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(false,false,true);
}
TEST_F(ChanceNavigation, CompleteMotionAndAlternativeStoppingPoliciesKeepSeparateUnchangedRiskLimits) {
  TerrainNavBenchmark::forward_cruising_is_not_the_whole_braking_policy();
}
TEST_F(ChanceNavigation, SafetyRootCannotBorrowTrackingLeadOrRefreshItsSourceStamp) {
  TerrainNavBenchmark::safety_root_is_current_while_tracking_lead_keeps_original_source_clock();
}
TEST_F(ChanceNavigation, StatePredictionAndAvoidanceShareIndependentActuatorDeadTimes) {
  TerrainNavBenchmark::control_root_uses_the_same_separate_actuator_delays_as_maneuver_proofs();
}
TEST_F(ChanceNavigation, QualifiedUnconfirmedHypothesisCanProveAnActionWithoutInventingPersistentConflictIdentity) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(false,false,false,true);
}
TEST_F(ChanceNavigation, DynamicActionsMustRespectTheSameConfiguredApproachSpeedAsNominalTracking) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(false,false,false,false,true);
}
TEST_F(ChanceNavigation, ARearTaskGoalCannotRemoveTheOnlyCertifiableForwardEscape) {
  TerrainNavBenchmark::dynamic_avoidance_prefers_certified_forward_progress(false,false,false,false,false,true);
}
TEST_F(ChanceNavigation, InteractionProgressUsesTheRealTurningRadiusAndTerminalHeadingWithoutChangingTerrain) {
  TerrainNavBenchmark::interaction_progress_accounts_for_the_actual_turning_radius_and_terminal_heading();
}
TEST_F(ChanceNavigation, RecedingInteractionCanImproveACertifiedArcButCannotReuseItAcrossMissingActorsOrUnknownTerrain) {
  TerrainNavBenchmark::receding_interaction_can_improve_a_still_safe_arc_without_reusing_its_old_permission();
}
TEST_F(ChanceNavigation, StoppedInteractionHandsBackButCannotClearUnknownSupportOrDiscardAFutureDrivePulse) {
  TerrainNavBenchmark::stopped_interaction_hands_back_without_renewing_its_stationary_tail_forever();
}
TEST_F(ChanceNavigation, LateMapReplacementReselectsWithFreshProofAndStopsOnRepeatedOrTotalRejection) {
  TerrainNavBenchmark::replaced_commit_inputs_require_a_new_bounded_action_proof();
}
TEST_F(ChanceNavigation, SelectionRefreshCannotUseExpiredSourcesOldRootOrForgetAnUnresolvedActor) {
  TerrainNavBenchmark::selection_refreshes_sources_and_actual_root_before_complete_proofs();
}
TEST_F(ChanceNavigation, PublicationBoundaryPredictionIsInvariantToAbsoluteClockOrigin) {
  TerrainNavBenchmark::absolute_clock_roundoff_cannot_invalidate_an_admitted_publication_boundary();
}
TEST_F(ChanceNavigation, LongerAnticipationCanActBeforeAShortHorizonMissesTheComingCrossing) {
  TerrainNavBenchmark::anticipation_can_start_a_certified_action_before_the_short_stop_becomes_unsafe();
}
TEST_F(ChanceNavigation, RemoteWeakCueCannotCancelMomentumButMissingConfirmationMustStop) {
  TerrainNavBenchmark::remote_weak_motion_is_warning_not_unconditional_braking();
}
TEST_F(ChanceNavigation, JointDynamicRiskMustShareStatePathAndBrakingLimitsWithTerrain) {
  TerrainNavBenchmark::joint_motion_probability_is_part_of_terrain_admission();
}
TEST_F(ChanceNavigation, OrdinaryBrakingCannotBypassCombinedRiskOrIntermediateCrossings) {
  TerrainNavBenchmark::ordinary_braking_shares_the_combined_budget_and_checks_the_whole_tail();
}
TEST_F(ChanceNavigation, ReusedImmediateFootprintQueryPreservesBothLedgersAndCannotSurviveANewInvalidGeneration) {
  TerrainNavBenchmark::footprint_query_reuse_keeps_the_same_pose_and_height_ledgers();
}
TEST_F(ChanceNavigation, DecisionGeometryCacheNeverCachesPolicyRiskOrCrossesImmutableInputGenerations) {
  TerrainNavBenchmark::decision_geometry_cache_preserves_policy_ledgers_and_input_generations();
}
TEST_F(ChanceNavigation, FairFamilyEvaluationCannotMakeASafeBrakeSuppressProvenForwardProgress) {
  TerrainNavBenchmark::fair_family_proofs_keep_better_forward_motion_and_safe_budget_fallback();
}
TEST_F(ChanceNavigation, KnownMotionSpeedPolicyCannotBypassFullCombinedCollisionAndBrakingProofs) {
  TerrainNavBenchmark::modeled_motion_speed_policy_never_grants_an_unsafe_command();
}
TEST_F(ChanceNavigation, EarlierStoppingUsesTheSameProbabilityModelAndChecksIntermediateCrossings) {
  TerrainNavBenchmark::earlier_stopping_guard_uses_the_configured_model_without_missing_crossing_times();
}
TEST_F(ChanceNavigation, PositiveConflictRetirementCannotInsertAnUnrequestedStopDuringNominalInputRecovery) {
  TerrainNavBenchmark::retiring_bookkeeping_preserves_a_fresh_policy_during_nominal_recovery();
}
TEST_F(ChanceNavigation, ExpiredRemoteMotionOrInvalidInputCannotInventOrEraseAConflict) {
  TerrainNavBenchmark::unavailable_motion_does_not_invent_or_erase_geometric_conflicts();
}
TEST_F(ChanceNavigation, MovingOccupancyOwnershipRequiresUniqueMeasuredSourceMatchedEvidence) {
  TerrainNavBenchmark::moving_ownership_is_unique_measured_and_source_exact();
}
TEST_F(ChanceNavigation, FutureActorSpaceCanBeUsedWithoutErasingWallsUnknownTerrainOrActualRootCollision) {
  TerrainNavBenchmark::moving_occupancy_uses_time_without_erasing_walls_unknown_or_actual_root();
}
TEST_F(ChanceNavigation, ConditionalConflictRetirementCannotForgetMissingTargetsOrCancelCertifiedMotion) {
  TerrainNavBenchmark::conditional_conflict_retirement_needs_positive_identity_but_not_unrelated_missing_objects();
}
TEST_F(ChanceNavigation, ExplicitObservationScopeDoesNotLatchMissingIdsOrEraseRawOccupancyOrIgnoreOutages) {
  TerrainNavBenchmark::observation_limited_motion_cannot_latch_missing_ids_or_clear_obstacles();
}
