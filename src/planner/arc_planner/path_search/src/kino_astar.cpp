#include <path_searching/kino_astar.h>
#include <algorithm>
#include <cmath>
#include <sstream>


#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tools/tic_toc.hpp>

using namespace std;
using namespace Eigen;

namespace path_searching
{
  KinoAstar::~KinoAstar()
  {
    for (auto *node : path_node_pool_)
    {
      delete node;
    }
    path_node_pool_.clear();
  }
   void KinoAstar::stateTransit(Eigen::Vector3d &state0,  Eigen::Vector3d &state1,
              Eigen::Vector2d &ctrl_input){
    //helpful var
    double psi = ctrl_input[0]; double s = ctrl_input[1]; 
    if(fabs(psi)>=0.01){
      double k = wheel_base/tan(psi);
      state1[0] = state0[0] + k*(sin(state0[2]+s/k)-sin(state0[2]));
      state1[1] = state0[1] - k*(cos(state0[2]+s/k)-cos(state0[2]));
      state1[2] = state0[2] + s/k;
    } 
    else{
      state1[0] = state0[0] + s * cos(state0[2]);
      state1[1] = state0[1] + s * sin(state0[2]);
      state1[2] = state0[2]; 
    }
  }
bool KinoAstar::isCollided(const Eigen::Vector3d &state){
  const Eigen::Vector3d query(state[0], state[1], 0.0);

  // Primary: use terrain risk layer from fused_map
  bool has_risk = false;
  double risk_val = 1.0;

  // Try risk_map_itf_ first, then frontend_map_itf_ (for backward compat)
  auto risk_map = risk_map_itf_ ? risk_map_itf_ : frontend_map_itf_;
  if (risk_map) {
    try {
      risk_val = risk_map->getValue("risk", query);
      has_risk = true;
    } catch (...) {}
  }

  if (has_risk) {
    // Check risk threshold — risk layer already encodes traversability.
    // NaN risk (unobserved) is treated as free (optimistic); safety gate catches issues.
    if (std::isfinite(risk_val) && risk_val >= risk_obstacle_threshold_) return true;
    return false;
  }

  // Fallback: use SDF if risk layer is not available (deprecated path)
  if (sdf_map_itf_) {
    try {
      double dis = sdf_map_itf_->getValue("sdf_cpu", query);
      return (dis <= safe_dis);
    } catch (...) {
      try {
        double dis = sdf_map_itf_->getValue("sdf", query);
        return (dis <= safe_dis);
      } catch (...) {}
    }
  }

  // No risk or SDF data available — default to free (optimistic for exploration)
  static int missing_layer_warn_count = 0;
  if (missing_layer_warn_count < 10)
  {
    std::cerr << "[WARN] No risk or sdf layer in isCollided(), default to free" << std::endl;
    ++missing_layer_warn_count;
  }
  return false;
}


int KinoAstar::search(Eigen::Vector4d start_state, Eigen::Vector2d init_ctrl,
                               Eigen::Vector4d end_state,double& model_time,bool use3d)
  {
    auto query_risk = [this](const Eigen::Vector3d &state) -> double {
      const Eigen::Vector3d query(state[0], state[1], 0.0);
      auto risk_map = risk_map_itf_ ? risk_map_itf_ : frontend_map_itf_;
      if (risk_map) {
        try {
          double r = risk_map->getValue("risk", query);
          if (std::isfinite(r)) return r;
        } catch (...) {}
      }
      // Fallback: try SDF (deprecated, kept for backward compat)
      if (sdf_map_itf_) {
        try {
          double dis = sdf_map_itf_->getValue("sdf_cpu", query);
          if (std::isfinite(dis))
            return (dis <= safe_dis) ? 1.0 : 0.0;
        } catch (...) {}
      }
      // No data available: treat as free (optimistic).
      // The safety gate and collision checker will catch real obstacles when data arrives.
      return 0.0;
    };

    auto project_to_free = [&](Eigen::Vector4d &state, const char *tag, bool seek_roomy = false) -> bool {
      const double clear_req_risk = seek_roomy ? risk_free_threshold_ * 0.5 : risk_free_threshold_;
      const double current_risk = query_risk(state.head(3));
      if (!seek_roomy && current_risk < clear_req_risk)
      {
        return true;  // already in free space
      }

      const Eigen::Vector2d origin = state.head(2);
      const double radius_max = 2.0;
      const double radius_step = std::max(0.08, resolution_);
      const int angle_samples = 36;
      const double target_risk = seek_roomy ? std::min(clear_req_risk, current_risk - 0.15) : clear_req_risk;
      double best_risk = current_risk;
      Eigen::Vector4d best_state = state;

      for (double r = radius_step; r <= radius_max + 1e-6; r += radius_step)
      {
        for (int k = 0; k < angle_samples; ++k)
        {
          const double ang = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(angle_samples);
          Eigen::Vector4d cand = state;
          cand[0] = origin[0] + r * std::cos(ang);
          cand[1] = origin[1] + r * std::sin(ang);
          const double risk = query_risk(cand.head(3));
          if (risk < best_risk)
          {
            best_risk = risk;
            best_state = cand;
          }
          if (risk < target_risk)
          {
            state = cand;
            std::cout << "[Kino Astar] projected " << tag << " to free space: ["
                      << state[0] << ", " << state[1] << "]"
                      << " (delta=" << r << ", risk=" << risk << ")" << std::endl;
            return true;
          }
        }
      }

      if (seek_roomy && best_risk < current_risk - 1e-3 && best_risk < risk_free_threshold_)
      {
        state = best_state;
        std::cout << "[Kino Astar] nudged " << tag << " to roomier space: ["
                  << state[0] << ", " << state[1] << "]"
                  << " (risk=" << best_risk << ", from=" << current_risk << ")" << std::endl;
        return true;
      }

      if (current_risk < clear_req_risk)
      {
        return true;
      }

      std::cout << "[Kino Astar] failed to project " << tag
                << " into free space (required risk<" << clear_req_risk << ")" << std::endl;
      return false;
    };

    model_time = 0.0;
    near_goal_time_extended_ = false;
    original_max_seach_time_ = max_seach_time;

    Eigen::Vector2d min_range, max_range;
    frontend_map_itf_->getMapRange(min_range, max_range);
    x_upper_ = max_range.x();
    y_upper_ = max_range.y();
    x_lower_ = min_range.x();
    y_lower_ = min_range.y();
    resolution_ = frontend_map_itf_->getResolutionPos();

    // Fallback: if the grid is empty (zero map range), use config horizon to
    // create a synthetic map window centered on the start position.
    const double map_span_x = x_upper_ - x_lower_;
    const double map_span_y = y_upper_ - y_lower_;
    if (map_span_x < 1.0 || map_span_y < 1.0 || resolution_ < 1e-6)
    {
        const double fallback_half = std::max(12.0, horizon_ * 1.5);
        x_lower_ = start_state[0] - fallback_half;
        x_upper_ = start_state[0] + fallback_half;
        y_lower_ = start_state[1] - fallback_half;
        y_upper_ = start_state[1] + fallback_half;
        resolution_ = std::max(0.1, resolution_);
        std::cout << "[Kino Astar] map range empty (span_x=" << map_span_x << " span_y=" << map_span_y
                  << "), using fallback centered on start: x[" << x_lower_ << "," << x_upper_
                  << "] y[" << y_lower_ << "," << y_upper_ << "] res=" << resolution_ << std::endl;
    }

    // Keep the planner's boundary states faithful to the caller.  The previous
    // behavior projected both start and goal whenever risk exceeded the very low
    // soft-risk threshold, which moved the trajectory origin and could also move
    // the mission goal before planning even started.  Only repair a goal that is
    // actually inside a hard obstacle; a colliding start must be rejected so the
    // FSM can decide how to recover from the real vehicle pose.
    if (isCollided(end_state.head(3)) && !project_to_free(end_state, "goal"))
    {
      return 0;
    }

    const Eigen::Vector2d vec_to_goal = end_state.head(2) - start_state.head(2);
    const double heading_to_goal = std::atan2(vec_to_goal.y(), vec_to_goal.x());
    const double heading_gap = std::fabs(normalize_angle(heading_to_goal - start_state[2]));
    const double goal_dist = vec_to_goal.norm();
    const bool goal_is_behind = (std::cos(heading_to_goal - start_state[2]) < -0.15) ||
                                (heading_gap > 1.75 && goal_dist > 0.8);
    // Only force pure-reverse for close goals. For distant goals, allow forward
    // arcs so the vehicle can turn around instead of reversing the entire way.
    // Only reverse straight to goal when it's very close behind (<3m).
    // For distant goals behind, the vehicle should turn around and drive forward.
    const bool close_behind = goal_is_behind && goal_dist < 2.0;  // only reverse for very close goals
    double search_forward_penalty = traj_forward_penalty;
    double search_back_penalty = traj_back_penalty;
    double search_gear_switch_penalty = traj_gear_switch_penalty;

    if (goal_is_behind && allow_reverse_search_)
    {
      if (close_behind)
      {
        // Very close goal behind: reversing is the right answer.
        // Make forward expensive and gear switches prohibitive.
        search_back_penalty = std::min(search_back_penalty, 2.0);
        search_gear_switch_penalty = std::max(search_gear_switch_penalty, 22.0);
        search_forward_penalty = std::max(1.5, search_forward_penalty);
      }
      else
      {
        // Distant goal behind (>=3m): must turn around, NOT reverse all the way.
        // The heuristic (lambda_heu_=5.0) gives reverse nodes much lower f_score
        // because they move directly toward the goal. We must make reverse SO
        // expensive that forward turn-around nodes win the f_score comparison.
        //   Reverse 1m: g=25, h=5*(d-1)  → f=5d+20
        //   Forward 1m: g=0.3, h=5*(d+1) → f=5d+5.3
        // Forward wins by ~14.7 per step → search explores turn-around path.
        search_back_penalty = 25.0;
        search_gear_switch_penalty = 0.2;
        search_forward_penalty = 0.3;
      }
      std::cout << "[Kino Astar] goal is behind start (dist=" << goal_dist
                << "m, close_behind=" << (close_behind ? "yes" : "no")
                << "), heading_gap=" << heading_gap
                << " fwd=" << search_forward_penalty
                << " back=" << search_back_penalty
                << " gear=" << search_gear_switch_penalty << std::endl;
    }

    double startTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    int expanded_num = 0;
    bool isocc = false;  bool initsearch = false;
    TicToc time_profile_tool_;
    time_profile_tool_.tic();
    const bool start_in_collision = isCollided(start_state.head(3));
    const double start_risk = query_risk(start_state.head(3));
    const double start_escape_radius = std::clamp(std::max(0.45, safe_dis + 0.10), 0.45, 0.95);
    if(start_in_collision){
      std::cerr << "[Kino Astar]: start in obstacle (risk=" << start_risk
                << ", risk_obs_thr=" << risk_obstacle_threshold_
                << "), allow bounded escape from actual pose" << std::endl;
    }
    if(isCollided(end_state.head(3))){
      const double goal_risk = query_risk(end_state.head(3));
      std::cerr << "[Kino Astar]: goal in obstacle (risk=" << goal_risk
                << ", risk_obs_thr=" << risk_obstacle_threshold_ << ")" << std::endl;
      return 0;
    }
    start_state_ = start_state;
    start_ctrl = init_ctrl;
    end_state_ = end_state;
    Eigen::Vector2i end_index;
    end_index = posToIndex(end_state.head(2));
    /* ---------- initialize ---------- */
    PathNodePtr cur_node = path_node_pool_[0];
    cur_node->parent = NULL;
    cur_node->state = start_state.head(3);
    cur_node->index = posToIndex(start_state.head(2));
    cur_node->yaw_idx = yawToIndex(start_state[2]);
    cur_node->g_score = 0.0;
    cur_node->input = Eigen::Vector2d(0.0,0.0);
    cur_node->singul = getSingularity(start_state(3));
    cur_node->f_score = lambda_heu_ * getHeu(cur_node->state, end_state);
    cur_node->node_state = IN_OPEN_SET;
    open_set_.push(cur_node);
    use_node_num_ += 1;
    if(!use3d)
      expanded_nodes_.insert(cur_node->index, cur_node);
    else
      expanded_nodes_.insert(cur_node->index, yawToIndex(start_state[2]),cur_node);
    PathNodePtr terminate_node = NULL;
    if(cur_node->singul == 0){ initsearch = true;}
    /* ---------- search loop ---------- */
    int out_of_range_total = 0;
    int collision_total = 0;
    int close_set_total = 0;
    int expanded_total = 0;
    double last_shot_try_dist = std::numeric_limits<double>::infinity();
    while (!open_set_.empty())
    {
      /* ---------- get lowest f_score node ---------- */
      cur_node = open_set_.top();

      /* ---------- determine termination ---------- */
      // to decide the near end
      bool reach_horizon = (cur_node->state.head(2) - start_state_.head(2)).norm() >= horizon_;
      double t1 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      const double cur_risk_for_escape = start_in_collision ? query_risk(cur_node->state) : 0.0;
      const bool cur_in_start_escape =
          start_in_collision &&
          std::isfinite(cur_risk_for_escape) &&
          cur_risk_for_escape >= risk_obstacle_threshold_ &&
          (cur_node->state.head(2) - start_state_.head(2)).norm() <= start_escape_radius;

      const Eigen::Vector2d shot_delta = end_state_.head(2) - cur_node->state.head(2);
      const double shot_dist = shot_delta.norm();
      if (enable_shot_ && shot_dist < shot_max_dist_ && !cur_in_start_escape)
      {
        const bool near_terminal = shot_dist < 2.5;
        const bool enough_new_progress =
            std::isinf(last_shot_try_dist) ||
            (last_shot_try_dist - shot_dist) > shot_progress_step_;
        if (near_terminal || enough_new_progress)
        {
          const double line_yaw = std::atan2(shot_delta.y(), shot_delta.x());
          const double fwd_gap = std::fabs(normalize_angle(line_yaw - cur_node->state[2]));
          const double rev_gap = std::fabs(normalize_angle(line_yaw + M_PI - cur_node->state[2]));
          const double heading_gap = allow_reverse_search_ ? std::min(fwd_gap, rev_gap) : fwd_gap;
          if (near_terminal || heading_gap < shot_heading_gate_)
          {
            last_shot_try_dist = shot_dist;
            is_shot_sucess(cur_node->state, end_state_.head(3));
          }
        }
      }
      double t2 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
      if (is_shot_succ_)
      {
        terminate_node = cur_node;
        retrievePath(terminate_node);
        has_path_ = true;
        if (is_shot_succ_)
        {
          /* one shot trajectory */
          std::cout << "one shot! iter num: " << iter_num_ << std::endl;
          double endTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
          std::cout << "search Time: "<<(endTime-startTime)*1000.0<<" ms" << std::endl;
          return 1;
        }
  
      }
      double runTime = time_profile_tool_.toc() / 1000.0;
      if(runTime > max_seach_time){
        terminate_node = cur_node;
        retrievePath(terminate_node);
        has_path_ = true;
        if (terminate_node->parent == NULL)
        {
          RCLCPP_ERROR(rclcpp::get_logger("kino_astar"),
                       "Kino Astar: terminate_node->parent == NULL");
          RCLCPP_ERROR(rclcpp::get_logger("kino_astar"), "Kino Astar: NO_PATH");
          return 0;
        }
        else
        {
          const double start_goal_dist = (start_state_.head(2) - end_state_.head(2)).norm();
          const double frontier_goal_dist = (terminate_node->state.head(2) - end_state_.head(2)).norm();
          const double frontier_travel = (terminate_node->state.head(2) - start_state_.head(2)).norm();
          const double progress = start_goal_dist - frontier_goal_dist;

          // Near-goal search extension: if the frontier is close to the goal
          // (< 2.5m) but hasn't reached it yet, extend the search time by
          // 50% to let A* find a kinematically sound terminal approach.
          // This avoids both stopping short AND creating twisted forced shots.
          if (!near_goal_time_extended_ && frontier_goal_dist < 2.5 &&
              frontier_goal_dist > 0.10 && frontier_travel > 0.5)
          {
              const double final_approach_factor = (start_goal_dist < 2.2) ? 2.2 : 1.5;
              max_seach_time = original_max_seach_time_ * final_approach_factor;
              near_goal_time_extended_ = true;
              std::cout << "[Kino Astar]: near-goal extension — search time increased to "
                        << max_seach_time << "s (frontier_dist=" << frontier_goal_dist << "m)" << std::endl;
              continue;  // keep searching with extended time
          }

          // In the final approach, do not publish a far frontier as a false
          // local goal.  A moderate remaining gap is still useful as rough
          // guidance because the backend keeps the true final state constraint.
          if (start_goal_dist < 2.2 && frontier_goal_dist > 1.25)
          {
            std::cout << "[Kino Astar]: final-approach partial frontier rejected"
                      << " (frontier_goal_dist=" << frontier_goal_dist
                      << ", start_goal_dist=" << start_goal_dist << ")" << std::endl;
            return 0;
          }

          // Relax required progress: continuous replanning can extend a useful
          // frontier when the mission goal is still far away.
          const double min_progress = 0.05;
          if (start_goal_dist > 1.2 &&
              (progress < min_progress || frontier_travel < 0.10))
          {
            std::cout << "[Kino Astar]: search timeout frontier rejected"
                      << " (progress=" << progress
                      << ", need=" << min_progress
                      << ", travel=" << frontier_travel << ")" << std::endl;
            return 0;
          }
          std::cout << "[Kino Astar]: search timeout, use partial frontier path. iter="
                    << iter_num_ << std::endl;
          return 1;
        }
      }
      /* ---------- pop node and add to close set ---------- */
      open_set_.pop();
      cur_node->node_state = IN_CLOSE_SET;
      /* ---------- init state propagation ---------- */
      Eigen::Vector3d cur_state = cur_node->state;
      Eigen::Vector3d pro_state;
      Eigen::Vector2d ctrl_input;
      vector<Eigen::Vector2d> inputs;
      double res = steer_res;
      for (double arc = -step_arc; arc <= step_arc + 1e-3; arc += 0.5*step_arc){
        if(fabs(arc)<0.05) continue;
        if (!allow_reverse_search_ && arc < -1e-3) continue;
        // Only block forward arcs when the goal is close behind — for distant
        // goals the vehicle should be able to turn around and drive forward.
        if (close_behind && arc > 1e-3)
          continue;
        
        for (double steer = -max_steer_; steer <= max_steer_ + 1e-3; steer += res * max_steer_*1.0)
        {
          ctrl_input << steer, arc;
          inputs.push_back(ctrl_input);
        }
      }
      /* ---------- state propagation loop ---------- */
      for (auto& input:inputs){
        int singul = input[1]>0?1:-1;
        stateTransit(cur_state, pro_state, input);
        iter_num_ += 1;

        /* inside map range */
        // Treat exact boundary points as valid to avoid rejecting edge-hugging motions.
        if (pro_state(0) < x_lower_ || pro_state(0) > x_upper_ ||
            pro_state(1) < y_lower_ || pro_state(1) > y_upper_)
        {
          ++out_of_range_total;
          static int out_of_range_log_count = 0;
          if (out_of_range_log_count < 3)
          {
            std::cout << "[Kino Astar]: out of map range, pos=(" << pro_state(0) << ", " << pro_state(1)
                      << ") bounds=x[" << x_lower_ << "," << x_upper_ << "] y[" << y_lower_ << "," << y_upper_ << "]" << endl;
            ++out_of_range_log_count;
          }
          continue;
        }
        /* not in close set */
        Eigen::Vector2i pro_id = posToIndex(pro_state.head(2));
        double pro_yaw_id = yawToIndex(pro_state[2]);
        PathNodePtr pro_node;
        if(use3d)
          pro_node = expanded_nodes_.find(pro_id, pro_yaw_id);
        else
          pro_node = expanded_nodes_.find(pro_id);
        if (pro_node != NULL && pro_node->node_state == IN_CLOSE_SET)
        {
          ++close_set_total;
          continue;
        }

        // /* not in the same voxel */
        Eigen::Vector2i diff = pro_id - cur_node->index;
        int diff_yaw = pro_yaw_id - cur_node->yaw_idx;
        if (diff.norm() == 0 && ((!use3d) || diff_yaw == 0))
        {
          continue;
        }
        /* collision free */
        Eigen::Vector3d xt;
        bool is_occ = false;
        for (int k = 1; k <= check_num_; ++k)
        {
          double tmparc = input[1] * double(k) / double(check_num_);
          Eigen::Vector2d tmpctrl; tmpctrl << input[0],tmparc;
          stateTransit(cur_state, xt, tmpctrl);
          is_occ = isCollided(xt);
              if (is_occ && start_in_collision)
              {
                const double escape_dist = (xt.head(2) - start_state_.head(2)).norm();
                const double xt_risk = query_risk(xt);
                // When the actual vehicle pose is already on a hard risk cell,
                // do not project the start away from the vehicle.  Instead,
                // allow only a short, monotonic escape band: the propagated
                // sample must stay within the bounded radius, must not be worse
                // than the original start risk, and after the first small move
                // must not climb relative to the current node.  This preserves a
                // real start pose while preventing the search from sliding along
                // obstacle boundaries.
                const bool immediate_escape_step =
                    escape_dist <= std::max(0.20, std::fabs(input[1]) / std::max(1, check_num_) + 0.05);
                const bool bounded_start_escape =
                    escape_dist <= start_escape_radius &&
                    std::isfinite(xt_risk) &&
                    xt_risk <= start_risk + 0.02 &&
                    (immediate_escape_step || !std::isfinite(cur_risk_for_escape) ||
                     xt_risk <= cur_risk_for_escape + 0.015);
                if (bounded_start_escape)
                  is_occ = false;
              }
	          if (is_occ)
	          {
	            break;
          }

        }
        if (is_occ)  { ++collision_total; continue; }

        /* ---------- compute cost ---------- */
        double tmp_g_score = 0.0;
        double tmp_f_score = 0.0;
        int lastDir = cur_node->singul;
        if(singul>0){
          tmp_g_score +=  std::fabs(input[1]) * search_forward_penalty;
        }
        else{
          tmp_g_score += std::fabs(input[1]) * search_back_penalty;
        }
        if(singul * lastDir < 0){
          tmp_g_score += search_gear_switch_penalty;
        }
        const double steer_ratio = std::abs(input[0]) / std::max(1e-3, max_steer_);
        tmp_g_score += traj_steer_penalty * std::fabs(input[0]) * std::fabs(input[1]);
        tmp_g_score += 0.22 * steer_ratio * steer_ratio * std::fabs(input[1]);
        tmp_g_score += traj_steer_change_penalty * std::fabs(input[0]-cur_node->input[0]) * 1.0;
        
        // Risk penalty: linear cost proportional to risk value.
        // Keeps the penalty gentle enough that the path only deviates when
        // risk is meaningful, avoiding the "zigzag bounce" of quadratic walls.
        double pro_risk = query_risk(pro_state.head(3));
        if (pro_risk > 0.005) {
            const double soft_risk = std::max(0.0, pro_risk - risk_free_threshold_);
            tmp_g_score +=
                (pro_risk * risk_linear_penalty_ +
                 soft_risk * soft_risk * risk_soft_penalty_) * std::fabs(input[1]);
        }

        // SDF distance penalty: gentle linear cost to nudge paths away from
        // obstacles.  Pure linear (no quadratic wall) to avoid bounce/zigzag.
        if (sdf_map_itf_) {
            try {
                double dis = sdf_map_itf_->getValue("sdf_cpu", Eigen::Vector3d(pro_state[0], pro_state[1], 0.0));
                if (std::isfinite(dis) && dis < 1.2) {
                    double margin = std::max(0.0, 1.2 - dis);
                    tmp_g_score += 20.0 * margin * std::fabs(input[1]);
                }
            } catch (...) {
                try {
                    double dis = sdf_map_itf_->getValue("sdf", Eigen::Vector3d(pro_state[0], pro_state[1], 0.0));
                    if (std::isfinite(dis) && dis < 1.2) {
                        double margin = std::max(0.0, 1.2 - dis);
                        tmp_g_score += 20.0 * margin * std::fabs(input[1]);
                    }
                } catch (...) {}
            }
        }

        // Direction guidance: penalize lateral deviation from previous trajectory direction.
        // This prevents the search from producing paths that go in a completely different
        // direction between consecutive replans, reducing "direction swing" behavior.
        if (has_prev_traj_direction_)
        {
            const Eigen::Vector2d node_to_goal = end_state_.head(2) - pro_state.head(2);
            const double dist_to_goal = node_to_goal.norm();
            if (dist_to_goal > 0.5)
            {
                const Eigen::Vector2d node_dir = node_to_goal / dist_to_goal;
                const double alignment = prev_traj_direction_.dot(node_dir);
                // Penalty: 0 when perfectly aligned, positive when pointing away.
                // Scale by remaining distance so far-away nodes get stronger guidance.
                const double lateral_penalty = direction_guidance_weight_ * (1.0 - alignment) * std::min(dist_to_goal, 12.0);
                tmp_g_score += lateral_penalty;
            }
        }
        tmp_g_score += cur_node->g_score;
        tmp_f_score = tmp_g_score + lambda_heu_ * getHeu(pro_state, end_state);
        /* ---------- compare expanded node in this loop ---------- */
        if (pro_node == NULL)
        {
          pro_node = path_node_pool_[use_node_num_];
          pro_node->index = pro_id;
          pro_node->state = pro_state;
          pro_node->yaw_idx = pro_yaw_id;
          pro_node->f_score = tmp_f_score;
          pro_node->g_score = tmp_g_score;
          pro_node->input = input;
          pro_node->parent = cur_node;
          pro_node->node_state = IN_OPEN_SET;
          pro_node->singul = singul;
          open_set_.push(pro_node);
          if(use3d)
            expanded_nodes_.insert(pro_id, pro_yaw_id, pro_node);
          else
            expanded_nodes_.insert(pro_id,pro_node);
          use_node_num_ += 1;
          if (use_node_num_ == allocate_num_)
          {
            cout << "run out of memory." << endl;
            return 0;
          }
        }
        else if (pro_node->node_state == IN_OPEN_SET)
        {

          if (tmp_g_score < pro_node->g_score)
          {
            pro_node->index = pro_id;
            pro_node->state = pro_state;
            pro_node->yaw_idx = pro_yaw_id;
            pro_node->f_score = tmp_f_score;
            pro_node->g_score = tmp_g_score;
            pro_node->input = input;
            pro_node->parent = cur_node;
            pro_node->singul = singul;
          }
        }
        else
        {
          cout << "error type in searching: " << pro_node->node_state << endl;
        }
      }
    }

    /* ---------- open set empty, no path ---------- */
    std::cout << "[Kino Astar] open set empty: expanded=" << expanded_total
              << " out_of_range=" << out_of_range_total
              << " collision=" << collision_total
              << " close_set=" << close_set_total
              << " iter=" << iter_num_ << std::endl;
    if (cur_node != NULL && cur_node->parent != NULL)
    {
      retrievePath(cur_node);
      has_path_ = true;
      std::cout << "[Kino Astar]: open set exhausted, fallback to partial path." << std::endl;
      return 1;
    }
    RCLCPP_ERROR(rclcpp::get_logger("kino_astar"), "open set empty, no path.");
    return 0;
  }


  bool KinoAstar::is_shot_sucess(Eigen::Vector3d state1,Eigen::Vector3d state2){
    
    std::vector<Eigen::Vector3d> path_list;
    double len;
    double ct1 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    computeShotTraj(state1,state2,path_list,len);
    double ct2 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    // std::cout<<"compute shot traj time: "<<(ct2-ct1)*1000.0<<" ms"<<std::endl;
    if (len > 0.25)
    {
      const double line_yaw = std::atan2(state2[1] - state1[1], state2[0] - state1[0]);
      auto heading_gap_to_line = [&](double yaw) {
        const double fwd_gap = std::fabs(normalize_angle(line_yaw - yaw));
        const double rev_gap = std::fabs(normalize_angle(line_yaw + M_PI - yaw));
        return allow_reverse_search_ ? std::min(fwd_gap, rev_gap) : fwd_gap;
      };

      const double start_gap = heading_gap_to_line(state1[2]);
      const double end_gap = heading_gap_to_line(state2[2]);
      const double yaw_span = std::fabs(normalize_angle(state2[2] - state1[2]));
      
      // For short terminal segments, accept the shot only when both endpoint
      // headings are also compatible with the chord.  Yaw span alone can accept
      // a side-slip-like final shortcut that the vehicle cannot track cleanly.
      if (len < 0.60 && yaw_span < 0.40 && start_gap < 0.45 && end_gap < 0.45) {
          is_shot_succ_ = true;
          return true;
      }

      const double denom = std::max(0.35, len);
      const double start_cur_req = 2.0 * std::sin(start_gap) / denom;
      const double end_cur_req = 2.0 * std::sin(end_gap) / denom;
      const double yaw_cur_req = yaw_span / denom;
      const double implied_cur = std::max(std::max(std::fabs(start_cur_req), std::fabs(end_cur_req)),
                                          std::fabs(yaw_cur_req));
      const double cur_limit = std::max(0.50, max_cur_ * 3.0);
      const bool extremely_close = len < 0.80;
      const double relaxed_cur_limit = extremely_close ? std::max(2.0, max_cur_ * 3.5) : cur_limit;
      const bool close_side_slip = len < 1.5 && std::max(start_gap, end_gap) > 0.60;
      
      if (start_gap > 0.80 || end_gap > 0.80 || implied_cur > relaxed_cur_limit || close_side_slip)
      {
            RCLCPP_DEBUG(
              rclcpp::get_logger("kino_astar"),
              "Reject one-shot: len=%.2f start_gap=%.2f end_gap=%.2f implied_cur=%.2f limit=%.2f",
              len, start_gap, end_gap, implied_cur, relaxed_cur_limit);
            return false;
      }
    }
    bool is_occ = false;
    double t1 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    for(unsigned int i = 0; i < path_list.size(); ++i){
        if(isCollided(path_list[i])) return false;
    }
    double t2 = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();


    is_shot_succ_ = true;
    return true;
  }



  void KinoAstar::computeShotTraj(Eigen::Vector3d &state1, Eigen::Vector3d &state2,
                                    std::vector<Eigen::Vector3d> &path_list,
                                    double& len){
    path_list.clear();
    const double dx = state2[0] - state1[0];
    const double dy = state2[1] - state1[1];
    len = std::hypot(dx, dy);

    const double yaw1 = normalize_angle(state1[2]);
    const double yaw2 = normalize_angle(state2[2]);
    double dyaw = yaw2 - yaw1;
    while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
    while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

    if (len < 1e-6)
    {
      path_list.push_back(Eigen::Vector3d(state1[0], state1[1], yaw1));
      path_list.push_back(Eigen::Vector3d(state2[0], state2[1], normalize_angle(yaw1 + dyaw)));
      return;
    }

    const int sample_num = std::max(2, static_cast<int>(std::ceil(len / std::max(1e-3, checkl))) + 1);
    for (int i = 0; i < sample_num; ++i)
    {
      const double r = static_cast<double>(i) / static_cast<double>(sample_num - 1);
      Eigen::Vector3d p;
      p[0] = state1[0] + r * dx;
      p[1] = state1[1] + r * dy;
      p[2] = normalize_angle(yaw1 + r * dyaw);
      path_list.push_back(p);
    }

    return;
  }
  // to retrieve the path to the correct order
  void KinoAstar::retrievePath(PathNodePtr end_node)
  {
    PathNodePtr cur_node = end_node;
    path_nodes_.push_back(cur_node);

    while (cur_node->parent != NULL)
    {
      cur_node = cur_node->parent;
      path_nodes_.push_back(cur_node);
    }

    reverse(path_nodes_.begin(), path_nodes_.end());
  }
  void KinoAstar::intialMap(std::shared_ptr<se2_grid::SE2Grid> map_itf)
  {
    sdf_map_itf_ = map_itf;
    // Only use SDF grid for map range if risk map is not available
    if (!frontend_map_itf_)
      frontend_map_itf_ = map_itf;
  }
  void KinoAstar::intialRiskMap(std::shared_ptr<se2_grid::SE2Grid> risk_map_itf)
  {
    risk_map_itf_ = risk_map_itf;
    // Also set frontend_map_itf_ for map range / resolution queries
    if (!frontend_map_itf_)
      frontend_map_itf_ = risk_map_itf;
  }
  void KinoAstar::init(ConfigPtr cfg_, std::shared_ptr<rclcpp::Node> node, bool un)
  {
    horizon_ = cfg_->horizon_;
    yaw_resolution_ = cfg_->yaw_resolution_;
    lambda_heu_ = cfg_->lambda_heu_;
    allocate_num_ = cfg_->allocate_num_;
    check_num_ = cfg_->check_num_;
    max_seach_time  = cfg_->max_seach_time;
    step_arc = cfg_->step_arc;
    checkl = cfg_->checkl;
    // Read risk-based collision thresholds from config
    // Lower clamp floors to allow conservative obstacle avoidance.
    // Gazebo live profile widens tree risk to radius=1.30m, power=2.0:
    //   d=0.75m -> risk≈0.18, d=1.0m -> risk≈0.05.
    // Hard keepout covers the inner band; this threshold catches high-risk cells.
    risk_obstacle_threshold_ = std::clamp(cfg_->risk_obstacle_threshold, 0.20, 1.0);
    risk_free_threshold_ = std::clamp(cfg_->risk_free_threshold, 0.02, 0.8);
    non_siguav = 0.0;
    wheel_base = cfg_->wheel_base;
    safe_dis = cfg_->safeMargin;
    /* ---------- pre-allocated node ---------- */
    for (auto *node_ptr : path_node_pool_)
    {
      delete node_ptr;
    }
    path_node_pool_.clear();
    path_node_pool_.resize(allocate_num_);
    for (int i = 0; i < allocate_num_; i++)
    {
      path_node_pool_[i] = new PathNode;
    }

    use_node_num_ = 0;
    iter_num_ = 0;
    yaw_origin_ = -M_PI;
    max_forward_vel = cfg_->vmax;
    max_forward_acc = cfg_->lonAccmax;
    max_backward_vel = max_forward_vel;
    max_backward_acc = max_forward_acc;
    max_cur_ = cfg_->kmax;
    traj_forward_penalty = 1.0;
    traj_back_penalty = std::max(4.0, cfg_->backW);
    traj_gear_switch_penalty = 30.0;
    std::cout << "max_cur_ = " << max_cur_ << std::endl;
    enable_shot_ = cfg_->enable_shot;
    max_steer_ = std::atan(wheel_base*max_cur_);
    inv_resolution_ = 1.0 / resolution_;
    inv_yaw_resolution_ = 1.0 / yaw_resolution_;
    steer_res = cfg_->steer_res;

    // ROS2 visualization publishers are created once and reused.
    if (node && rclcpp::ok(node->get_node_base_interface()->get_context()))
    {
      traj_forward_penalty = node->declare_parameter<double>("front_end/traj_forward_penalty", traj_forward_penalty);
      traj_back_penalty = node->declare_parameter<double>("front_end/traj_back_penalty", traj_back_penalty);
      traj_gear_switch_penalty = node->declare_parameter<double>("front_end/traj_gear_switch_penalty", traj_gear_switch_penalty);
      traj_steer_penalty = node->declare_parameter<double>("front_end/traj_steer_penalty", traj_steer_penalty);
      traj_steer_change_penalty = node->declare_parameter<double>("front_end/traj_steer_change_penalty", traj_steer_change_penalty);
      check_num_ = node->declare_parameter<int>("front_end/collision_check_num", check_num_);
      allow_reverse_search_ = node->declare_parameter<bool>("front_end/allow_reverse", false);
      shot_max_dist_ = node->declare_parameter<double>("front_end/shot_max_dist", shot_max_dist_);
      shot_heading_gate_ = node->declare_parameter<double>("front_end/shot_heading_gate", shot_heading_gate_);
      shot_progress_step_ = node->declare_parameter<double>("front_end/shot_progress_step", shot_progress_step_);
      risk_linear_penalty_ = node->declare_parameter<double>("front_end/risk_linear_penalty", risk_linear_penalty_);
      risk_soft_penalty_ = node->declare_parameter<double>("front_end/risk_soft_penalty", risk_soft_penalty_);

      check_num_ = std::max(2, check_num_);
      traj_steer_penalty = std::max(0.0, traj_steer_penalty);
      traj_steer_change_penalty = std::max(0.0, traj_steer_change_penalty);
      risk_linear_penalty_ = std::clamp(risk_linear_penalty_, 0.0, 160.0);
      risk_soft_penalty_ = std::clamp(risk_soft_penalty_, 0.0, 600.0);
      shot_max_dist_ = std::clamp(shot_max_dist_, 2.5, std::max(2.5, horizon_));
      shot_heading_gate_ = std::clamp(shot_heading_gate_, 0.35, 1.45);
      shot_progress_step_ = std::clamp(shot_progress_step_, 0.20, 2.00);

      if (!expandNodesVis)
      {
        expandNodesVis = node->create_publisher<sensor_msgs::msg::PointCloud2>("/vis/expanded_nodes", 1);
      }
      if (!networkVis)
      {
        networkVis = node->create_publisher<sensor_msgs::msg::PointCloud2>("/vis/neuralAstarMap", 1);
      }

      RCLCPP_INFO(node->get_logger(),
                  "KinoAstar search params: allow_reverse=%d, penalty(fwd/back/switch/steer/dsteer)=%.2f/%.2f/%.2f/%.2f/%.2f, risk(linear/soft)=%.1f/%.1f, check_num=%d, shot=(max=%.1fm gate=%.2frad step=%.2fm)",
                  allow_reverse_search_, traj_forward_penalty, traj_back_penalty,
                  traj_gear_switch_penalty, traj_steer_penalty, traj_steer_change_penalty,
                  risk_linear_penalty_, risk_soft_penalty_,
                  check_num_, shot_max_dist_, shot_heading_gate_, shot_progress_step_);
    }
  }

  void KinoAstar::warm(){
        return;
  }
    
  void KinoAstar::reset()
  {
    expanded_nodes_.clear();
    path_nodes_.clear();

    std::priority_queue<PathNodePtr, std::vector<PathNodePtr>, NodeComparator> empty_queue;
    open_set_.swap(empty_queue);

    for (int i = 0; i < use_node_num_; i++)
    {
      PathNodePtr node = path_node_pool_[i];
      node->parent = NULL;
      node->node_state = NOT_EXPAND;
    }

    use_node_num_ = 0;
    iter_num_ = 0;
    is_shot_succ_ = false;
    near_goal_time_extended_ = false;

  }
  std::vector<Eigen::Vector3d> KinoAstar::getSampleTraj(){
    return SampleTraj;
  }
  Eigen::Vector3d KinoAstar::evaluatePos(double t){
    if (SampleTraj.empty())
      return Eigen::Vector3d::Zero();
    if (SampleTraj.size() == 1 || shot_timeList.empty() ||
        shot_lengthList.size() != shot_timeList.size() ||
        shot_SList.size() != shot_timeList.size() ||
        shotindex.size() != shot_timeList.size() + 1)
    {
      return SampleTraj.back();
    }

    double total_time = 0.0;
    for (const double dt : shot_timeList)
    {
      if (!std::isfinite(dt) || dt <= 1.0e-5)
        return SampleTraj.back();
      total_time += dt;
    }
    if (!std::isfinite(total_time) || total_time <= 1.0e-5)
      return SampleTraj.back();

    if (!std::isfinite(t))
      t = 0.0;
    t = std::min<double>(std::max<double>(0.0, t), total_time);
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
    int index = -1;
    double tmpT = 0.0;
    double CutTime = 0.0;
    //locate the local traj
    for(int i = 0;i<shot_timeList.size();i++){
      const double seg_time = shot_timeList[i];
      if(tmpT + seg_time >= t - 1.0e-9) {
        index = i;
        CutTime = std::min(std::max(t - tmpT, 0.0), seg_time);
        break;
      }
      tmpT += seg_time;
    }
    if (index < 0)
      return SampleTraj.back();

    double initv = non_siguav,finv = non_siguav;
    if(index==0) {initv  = startvel;}
    if(index==shot_lengthList.size() - 1) finv = endvel;
    double localtime = shot_timeList[index];
    double locallength = shot_lengthList[index];
    if (!std::isfinite(localtime) || !std::isfinite(locallength) ||
        localtime <= 1.0e-5 || locallength <= 1.0e-5)
      return SampleTraj.back();
    int front = shotindex[index];
    int back =  shotindex[index+1];
    if (front < 0 || back < front || back >= static_cast<int>(SampleTraj.size()))
      return SampleTraj.back();
    std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+front,SampleTraj.begin()+back+1);
    if (localTraj.size() < 2)
      return localTraj.empty() ? SampleTraj.back() : localTraj.back();
    //find the nearest point
    double arclength;
    if(shot_SList[index] > 0)
      arclength= evaluateLength(CutTime,locallength,localtime,max_forward_vel,max_forward_acc, initv,finv);
    else
      arclength= evaluateLength(CutTime,locallength,localtime,max_backward_vel, max_backward_acc, initv,finv);
    if (!std::isfinite(arclength))
      arclength = locallength;
    arclength = std::min(std::max(arclength, 0.0), locallength);
    double tmparc = 0;
    for(int i = 0; i < static_cast<int>(localTraj.size())-1;i++){
      const double l = (localTraj[i+1]-localTraj[i]).head(2).norm();
      if (!std::isfinite(l) || l < 1.0e-6)
        continue;
      tmparc += l;
      if(tmparc>=arclength){
        double l1 = tmparc-arclength;
        double l2 = l-l1;//l2
        Eigen::Vector3d state = l1/l*localTraj[i]+l2/l*localTraj[i+1];
        if(fabs(localTraj[i+1][2]-localTraj[i][2])>=M_PI){
          double normalize_yaw = state[2];
          if(localTraj[i+1][2]<=0){
            normalize_yaw = l1/l*localTraj[i][2]+l2/l*(localTraj[i+1][2]+2*M_PI);
          }
          else if(localTraj[i][2]<=0){
            normalize_yaw = l1/l*(localTraj[i][2]+2*M_PI)+l2/l*localTraj[i+1][2];
          }
          state[2] = normalize_angle(normalize_yaw);
        }
        if (std::isfinite(state.x()) && std::isfinite(state.y()) && std::isfinite(state.z()))
          return state;
        return localTraj.back();
      }
    }
    return localTraj.back();
  }
  std::vector<Eigen::Vector4d> KinoAstar::SamplePosList(int N){
      double dt = totalTrajTime/N;
      std::vector<Eigen::Vector4d>  path;
      for(int i=0;i<=N;i++){
        double time = i*dt;
        Eigen::Vector4d posAndt;
        posAndt<<evaluatePos(time),time;
        path.push_back(posAndt);
      }
      return path;
  }
std::vector<Eigen::Vector3d> KinoAstar::getRoughSamples(){

    double truncate_len = 25.0;
    bool exceed_len = false;
    std::vector<Eigen::Vector3d> roughSampleList;
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
    PathNodePtr node = path_nodes_.back();
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;
    while(node->parent != NULL){
      for (int k = check_num_; k >0; k--)
      {
        Eigen::Vector3d state;
        double tmparc = node->input[1] * double(k) / double(check_num_);
        Eigen::Vector2d tmpctrl; tmpctrl << node->input[0],tmparc;
        stateTransit(node->parent->state, state, tmpctrl);
        state[2] = normalize_angle(state[2]);
        roughSampleList.push_back(state);

      }
      node = node->parent;
    } 
    start_state_[2] = normalize_angle(start_state_[2]);
    roughSampleList.push_back(start_state_.head(3));
    reverse(roughSampleList.begin(),roughSampleList.end());

    if(is_shot_succ_){
      std::vector<Eigen::Vector3d> shot_path;
      Eigen::Vector3d state1 = roughSampleList.back();
      Eigen::Vector3d state2 = end_state_.head(3);
      double shotLength = 0.0;
      computeShotTraj(state1, state2, shot_path, shotLength);
      for (size_t i = 1; i < shot_path.size(); ++i)
      {
        shot_path[i][2] = normalize_angle(shot_path[i][2]);
        roughSampleList.push_back(shot_path[i]);
      }
    }
    //truncate the init trajectory
    double tmp_len = 0;
    int truncate_idx = 0;
    for(truncate_idx = 0;truncate_idx <roughSampleList.size()-1;truncate_idx++){  
      tmp_len += (roughSampleList[truncate_idx+1]-roughSampleList[truncate_idx]).norm();
    }
    roughSampleList.assign(roughSampleList.begin(),roughSampleList.begin()+truncate_idx+1);
    SampleTraj = roughSampleList;
    return SampleTraj;
  }

  void KinoAstar::getKinoNode(KinoTrajData &flat_trajs)
  {
    double truncate_len = 25.0;
    bool exceed_len = false;
    flat_trajs.clear();
    std::vector<Eigen::Vector3d> roughSampleList;
    double startvel = fabs(start_state_[3]);
    double endvel = fabs(end_state_[3]);
    PathNodePtr node = path_nodes_.back();
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;
    while(node->parent != NULL){
      for (int k = check_num_; k >0; k--)
      {
        Eigen::Vector3d state;
        double tmparc = node->input[1] * double(k) / double(check_num_);
        Eigen::Vector2d tmpctrl; tmpctrl << node->input[0],tmparc;
        stateTransit(node->parent->state, state, tmpctrl);
        state[2] = normalize_angle(state[2]);
        roughSampleList.push_back(state);

      }
      node = node->parent;
    } 
    start_state_[2] = normalize_angle(start_state_[2]);
    roughSampleList.push_back(start_state_.head(3));
    reverse(roughSampleList.begin(),roughSampleList.end());

    if(is_shot_succ_){
      std::vector<Eigen::Vector3d> shot_path;
      Eigen::Vector3d state1 = roughSampleList.back();
      Eigen::Vector3d state2 = end_state_.head(3);
      double shotLength = 0.0;
      computeShotTraj(state1, state2, shot_path, shotLength);
      for (size_t i = 1; i < shot_path.size(); ++i)
      {
        shot_path[i][2] = normalize_angle(shot_path[i][2]);
        roughSampleList.push_back(shot_path[i]);
      }
    }
    //truncate the init trajectory
    double tmp_len = 0;
    int truncate_idx = 0;
    for(truncate_idx = 0;truncate_idx <roughSampleList.size()-1;truncate_idx++){  
      tmp_len += (roughSampleList[truncate_idx+1]-roughSampleList[truncate_idx]).norm();
    }
    roughSampleList.assign(roughSampleList.begin(),roughSampleList.begin()+truncate_idx+1);
    SampleTraj = roughSampleList;
    /*divide the whole shot traj into different segments*/   
    shot_lengthList.clear();
    shot_timeList.clear();
    shotindex.clear();
    shot_SList.clear(); 
    double tmpl = 0;
    bool ifnewtraj = false;
    int lastS = (SampleTraj[1]-SampleTraj[0]).head(2).dot(Eigen::Vector2d(cos(SampleTraj[0][2]),sin(SampleTraj[0][2])))>=0?1:-1;
    shotindex.push_back(0);
    for(int i = 0; i<SampleTraj.size()-1; i++){
      Eigen::Vector3d state1 = SampleTraj[i];
      Eigen::Vector3d state2 = SampleTraj[i+1];
      int curS = (state2-state1).head(2).dot(Eigen::Vector2d(cos(state1[2]),sin(state1[2]))) >=0 ? 1:-1;
      if(curS*lastS >= 0){
        tmpl += (state2-state1).head(2).norm();
      }
      else{  
        shotindex.push_back(i);
        shot_SList.push_back(lastS);
        const double seg_len = std::max(tmpl, 1.0e-4);
        shot_lengthList.push_back(seg_len);
        if(lastS>0)
          shot_timeList.push_back(evaluateDuration(seg_len,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
        else
          shot_timeList.push_back(evaluateDuration(seg_len,max_backward_vel, max_backward_acc, non_siguav,non_siguav));

        tmpl = (state2-state1).head(2).norm();
      }       
      lastS = curS;
    }
    shot_SList.push_back(lastS);
    const double last_seg_len = std::max(tmpl, 1.0e-4);
    shot_lengthList.push_back(last_seg_len);
    if(lastS>0)
      shot_timeList.push_back(evaluateDuration(last_seg_len,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
    else
      shot_timeList.push_back(evaluateDuration(last_seg_len,max_backward_vel, max_backward_acc, non_siguav,non_siguav));
    shotindex.push_back(SampleTraj.size()-1);
    if(shot_timeList.size()>=2){
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,non_siguav);
      else
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel, max_backward_acc, startvel,non_siguav);
      if(shot_SList[shot_timeList.size()-1]>0)
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_forward_vel,max_forward_acc,non_siguav,endvel);
      else
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_backward_vel,max_backward_acc,non_siguav,endvel);
    }
	    else{
	      if(shot_SList[0]>0)
	        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,endvel);
	      else  
	        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel,max_backward_acc, startvel,endvel);
	    }
	    for (size_t i = 0; i < shot_lengthList.size(); ++i)
	    {
	      if (!std::isfinite(shot_lengthList[i]) || !std::isfinite(shot_timeList[i]) ||
	          shot_lengthList[i] <= 1.0e-5 || shot_timeList[i] <= 1.0e-5 ||
	          shotindex[i] < 0 || shotindex[i + 1] <= shotindex[i] ||
	          shotindex[i + 1] >= static_cast<int>(SampleTraj.size()))
	      {
	        std::cerr << "[Kino Astar]: invalid flat-traj segment"
	                  << " idx=" << i
	                  << " len=" << shot_lengthList[i]
	                  << " time=" << shot_timeList[i]
	                  << " front=" << shotindex[i]
	                  << " back=" << shotindex[i + 1] << std::endl;
	        shot_lengthList.clear();
	        shot_timeList.clear();
	        shot_SList.clear();
	        shotindex.clear();
	        flat_trajs.clear();
	        return;
	      }
	    }
	    /*extract flat traj  
	    the flat traj include the end point but not the first point*/
	    for(int i=0;i<shot_lengthList.size();i++){
	      double initv = non_siguav,finv = non_siguav;
	      Eigen::Vector2d Initctrlinput,Finctrlinput;
      Initctrlinput<<0,0;Finctrlinput<<0,0;
      if(i==0) {initv  = startvel; Initctrlinput = start_ctrl;}
      if(i==shot_lengthList.size() - 1) finv = endvel;

      double locallength = shot_lengthList[i];
      int sig = shot_SList[i];
      std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+shotindex[i],SampleTraj.begin()+shotindex[i+1]+1);
      traj_pts.clear();
      thetas.clear();        
      double samplet;
      double tmparc = 0;
      int index = 0;
      double sampletime = 0.1;
      if(shot_timeList[i]<=sampletime){
        sampletime = shot_timeList[i] / 2.0;
      }
      for(samplet = sampletime; samplet<shot_timeList[i]; samplet+=sampletime){
        double arc;
        if(sig > 0){
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_forward_vel, max_forward_acc, initv, finv);
        }
        else{
          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_backward_vel, max_backward_acc, initv, finv);
        }
        //find the nearest point
        for(int k = index; k<localTraj.size()-1 ;k++)
        {
          tmparc+= (localTraj[k+1]-localTraj[k]).head(2).norm();
          if(tmparc>=arc){
            index = k;
            double l1 = tmparc-arc;
            double l = (localTraj[k+1]-localTraj[k]).head(2).norm();
            double l2 = l-l1;//l2
            double px = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[0];
            double py = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[1];
            double yaw= (l1/l*localTraj[k]+l2/l*localTraj[k+1])[2];
            if(fabs(localTraj[k+1][2]-localTraj[k][2])>=M_PI){   
              double normalize_yaw;
              if(localTraj[k+1][2]<=0){
                normalize_yaw = l1/l*localTraj[k][2]+l2/l*(localTraj[k+1][2]+2*M_PI);
              }
              else if(localTraj[k][2]<=0){
                normalize_yaw = l1/l*(localTraj[k][2]+2*M_PI)+l2/l*localTraj[k+1][2];
              }
              yaw = normalize_yaw;
            }
            traj_pts.push_back(Eigen::Vector3d(px,py,sampletime));
            thetas.push_back(yaw);
            tmparc -=(localTraj[k+1]-localTraj[k]).head(2).norm();
            break;
          }
        }      
      }
      traj_pts.push_back(Eigen::Vector3d(localTraj.back()[0],localTraj.back()[1],shot_timeList[i]-(samplet-sampletime)));
      thetas.push_back(localTraj.back()[2]);
      FlatTrajData flat_traj;
      Eigen::MatrixXd startS;
      Eigen::MatrixXd endS;
      getFlatState(Eigen::Vector4d(localTraj.front()[0],localTraj.front()[1],localTraj.front()[2],initv),Initctrlinput,startS,sig);
      getFlatState(Eigen::Vector4d(localTraj.back()[0],localTraj.back()[1],localTraj.back()[2],finv),Finctrlinput,endS,sig);
      flat_traj.traj_pts = traj_pts;
      flat_traj.thetas = thetas;
      flat_traj.start_state = startS;
      flat_traj.final_state = endS;
      flat_traj.singul = sig;
      flat_traj.duration = shot_timeList[i];
      flat_trajs.push_back(flat_traj);
    }
    totalTrajTime = 0.0;
    for(const auto dt : shot_timeList){
       totalTrajTime += dt; 
    }
  }
  
  void KinoAstar::getKinoNode(KinoTrajData &flat_trajs, std::vector<Eigen::Vector3d> inputSamples){
    double truncate_len = 25.0;
    bool exceed_len = false;

    flat_trajs.clear();
    double startvel = 0.0;
    double endvel = 0.0;
    std::vector<Eigen::Vector3d> traj_pts;  // 3, N
    std::vector<double> thetas;
    Eigen::Vector4d x0, xt;
    Vector2d ut, u0;
    SampleTraj.clear();
    SampleTraj.reserve(inputSamples.size());
    for (const auto &p : inputSamples)
    {
      if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
        continue;
      if (!SampleTraj.empty() && (p.head(2) - SampleTraj.back().head(2)).norm() < 1.0e-5)
      {
        SampleTraj.back()(2) = p(2);
        continue;
      }
      SampleTraj.push_back(p);
    }
    if (SampleTraj.size() < 2)
    {
      std::cerr << "[Kino Astar]: getKinoNode skipped degenerate sample trajectory"
                << std::endl;
      return;
    }
    /*divide the whole shot traj into different segments*/   
    shot_lengthList.clear();
    shot_timeList.clear();
    shotindex.clear();
    shot_SList.clear(); 
    double tmpl = 0;
    bool ifnewtraj = false;
    int lastS = (SampleTraj[1]-SampleTraj[0]).head(2).dot(Eigen::Vector2d(cos(SampleTraj[0][2]),sin(SampleTraj[0][2])))>=0?1:-1;
    shotindex.push_back(0);
    for(int i = 0; i<SampleTraj.size()-1; i++){
      Eigen::Vector3d state1 = SampleTraj[i];
      Eigen::Vector3d state2 = SampleTraj[i+1];
      int curS = (state2-state1).head(2).dot(Eigen::Vector2d(cos(state1[2]),sin(state1[2]))) >=0 ? 1:-1;
      if(curS*lastS >= 0){
        tmpl += (state2-state1).head(2).norm();
      }
      else{  
        shotindex.push_back(i);
        shot_SList.push_back(lastS);
        const double seg_len = std::max(tmpl, 1.0e-4);
        shot_lengthList.push_back(seg_len);
        if(lastS>0)
          shot_timeList.push_back(evaluateDuration(seg_len,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
        else
          shot_timeList.push_back(evaluateDuration(seg_len,max_backward_vel, max_backward_acc, non_siguav,non_siguav));

        tmpl = (state2-state1).head(2).norm();
      }       
      lastS = curS;
    }
    shot_SList.push_back(lastS);
    const double last_seg_len = std::max(tmpl, 1.0e-4);
    shot_lengthList.push_back(last_seg_len);
    if(lastS>0)
      shot_timeList.push_back(evaluateDuration(last_seg_len,max_forward_vel, max_forward_acc, non_siguav,non_siguav));
    else
      shot_timeList.push_back(evaluateDuration(last_seg_len,max_backward_vel, max_backward_acc, non_siguav,non_siguav));
    shotindex.push_back(SampleTraj.size()-1);
    if (shot_lengthList.size() != shot_timeList.size() ||
        shot_lengthList.size() != shot_SList.size() ||
        shotindex.size() != shot_lengthList.size() + 1)
    {
      std::cerr << "[Kino Astar]: inconsistent flat-traj segment buffers"
                << " len=" << shot_lengthList.size()
                << " time=" << shot_timeList.size()
                << " sign=" << shot_SList.size()
                << " index=" << shotindex.size() << std::endl;
      shot_lengthList.clear();
      shot_timeList.clear();
      shot_SList.clear();
      shotindex.clear();
      return;
    }
    if(shot_timeList.size()>=2){
      if(shot_SList[0]>0)
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,non_siguav);
      else
        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel, max_backward_acc, startvel,non_siguav);
      if(shot_SList[shot_timeList.size()-1]>0)
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_forward_vel,max_forward_acc,non_siguav,endvel);
      else
        shot_timeList[shot_timeList.size()-1] = evaluateDuration(shot_lengthList.back(),max_backward_vel,max_backward_acc,non_siguav,endvel);
    }
	    else{
	      if(shot_SList[0]>0)
	        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_forward_vel,max_forward_acc, startvel,endvel);
	      else  
	        shot_timeList[0] = evaluateDuration(shot_lengthList[0],max_backward_vel,max_backward_acc, startvel,endvel);
	    }
	    for (size_t i = 0; i < shot_lengthList.size(); ++i)
	    {
	      if (!std::isfinite(shot_lengthList[i]) || !std::isfinite(shot_timeList[i]) ||
	          shot_lengthList[i] <= 1.0e-5 || shot_timeList[i] <= 1.0e-5 ||
	          shotindex[i] < 0 || shotindex[i + 1] <= shotindex[i] ||
	          shotindex[i + 1] >= static_cast<int>(SampleTraj.size()))
	      {
	        std::cerr << "[Kino Astar]: invalid input-sample flat segment"
	                  << " idx=" << i
	                  << " len=" << shot_lengthList[i]
	                  << " time=" << shot_timeList[i]
	                  << " front=" << shotindex[i]
	                  << " back=" << shotindex[i + 1] << std::endl;
	        shot_lengthList.clear();
	        shot_timeList.clear();
	        shot_SList.clear();
	        shotindex.clear();
	        flat_trajs.clear();
	        return;
	      }
	    }
	    /*extract flat traj  
	    the flat traj include the end point but not the first point*/
	    for(int i=0;i<shot_lengthList.size();i++){
      double initv = non_siguav,finv = non_siguav;
      Eigen::Vector2d Initctrlinput,Finctrlinput;
      Initctrlinput<<0,0;Finctrlinput<<0,0;
      if(i==0) {initv  = startvel; Initctrlinput = start_ctrl;}
      if(i==shot_lengthList.size() - 1) finv = endvel;

	      double locallength = shot_lengthList[i];
	      int sig = shot_SList[i];
	      std::vector<Eigen::Vector3d> localTraj;localTraj.assign(SampleTraj.begin()+shotindex[i],SampleTraj.begin()+shotindex[i+1]+1);
	      if (localTraj.size() < 2)
	        continue;
	      traj_pts.clear();
	      thetas.clear();        
	      double samplet;
	      double tmparc = 0;
	      int index = 0;
	      double sampletime = 0.1;
	      if(shot_timeList[i]<=sampletime){
	        sampletime = shot_timeList[i] / 2.0;
	      }
	      if (!std::isfinite(sampletime) || sampletime <= 1.0e-5)
	        continue;
	      for(samplet = sampletime; samplet<shot_timeList[i]; samplet+=sampletime){
	        double arc;
	        if(sig > 0){
	          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_forward_vel, max_forward_acc, initv, finv);
	        }
	        else{
	          arc = evaluateLength(samplet,locallength,shot_timeList[i],max_backward_vel, max_backward_acc, initv, finv);
	        }
	        if (!std::isfinite(arc))
	          continue;
	        arc = std::min(std::max(arc, 0.0), locallength);
	        //find the nearest point
	        for(int k = index; k<localTraj.size()-1 ;k++)
	        {
	          const double l = (localTraj[k+1]-localTraj[k]).head(2).norm();
	          if (!std::isfinite(l) || l < 1.0e-6)
	            continue;
	          tmparc+= l;
	          if(tmparc>=arc){
	            index = k;
	            double l1 = tmparc-arc;
	            double l2 = l-l1;//l2
	            double px = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[0];
	            double py = (l1/l*localTraj[k]+l2/l*localTraj[k+1])[1];
	            double yaw= (l1/l*localTraj[k]+l2/l*localTraj[k+1])[2];
            if(fabs(localTraj[k+1][2]-localTraj[k][2])>=M_PI){   
              double normalize_yaw;
              if(localTraj[k+1][2]<=0){
                normalize_yaw = l1/l*localTraj[k][2]+l2/l*(localTraj[k+1][2]+2*M_PI);
              }
              else if(localTraj[k][2]<=0){
                normalize_yaw = l1/l*(localTraj[k][2]+2*M_PI)+l2/l*localTraj[k+1][2];
              }
              yaw = normalize_yaw;
            }
	            if (std::isfinite(px) && std::isfinite(py) && std::isfinite(yaw))
	            {
	              traj_pts.push_back(Eigen::Vector3d(px,py,sampletime));
	              thetas.push_back(normalize_angle(yaw));
	            }
	            tmparc -= l;
	            break;
	          }
	        }      
	      }
	      double tail_dt = shot_timeList[i]-(samplet-sampletime);
	      if (!std::isfinite(tail_dt) || tail_dt <= 1.0e-5)
	        tail_dt = std::max(shot_timeList[i], 1.0e-4);
	      if (std::isfinite(localTraj.back()[0]) && std::isfinite(localTraj.back()[1]) &&
	          std::isfinite(localTraj.back()[2]))
	      {
	        traj_pts.push_back(Eigen::Vector3d(localTraj.back()[0],localTraj.back()[1],tail_dt));
	        thetas.push_back(normalize_angle(localTraj.back()[2]));
	      }
	      if (traj_pts.empty() || traj_pts.size() != thetas.size())
	        continue;
	      FlatTrajData flat_traj;
      Eigen::MatrixXd startS;
      Eigen::MatrixXd endS;
      getFlatState(Eigen::Vector4d(localTraj.front()[0],localTraj.front()[1],localTraj.front()[2],initv),Initctrlinput,startS,sig);
      getFlatState(Eigen::Vector4d(localTraj.back()[0],localTraj.back()[1],localTraj.back()[2],finv),Finctrlinput,endS,sig);
      flat_traj.traj_pts = traj_pts;
      flat_traj.thetas = thetas;
      flat_traj.start_state = startS;
      flat_traj.final_state = endS;
      flat_traj.singul = sig;
      flat_traj.duration = shot_timeList[i];
      flat_trajs.push_back(flat_traj);
    }
    totalTrajTime = 0.0;
    for(const auto dt : shot_timeList){
       totalTrajTime += dt; 
    }
  }
  double KinoAstar::evaluateDuration(double length, double max_vel, double max_acc, double startV, double endV){
   double critical_len; //the critical length of two-order optimal control, acc is the input
    if(startV>max_vel||endV>max_vel){
      std::cerr << "start or end vel is larger than max_vel" << std::endl;
    }
    double startv2 = pow(startV,2);
    double endv2 = pow(endV,2);
    double maxv2 = pow(max_vel,2);
    critical_len = (maxv2-startv2)/(2*max_acc)+(maxv2-endv2)/(2*max_acc);
    if(length>=critical_len){
      return (max_vel-startV)/max_acc+(max_vel-endV)/max_acc+(length-critical_len)/max_vel;
    }
    else{
      double tmpv = sqrt(0.5*(startv2+endv2+2*max_acc*length));
      return (tmpv-startV)/max_acc + (tmpv-endV)/max_acc;
    }


  }
  double KinoAstar::evaluateLength(double curt,double locallength,double localtime,double max_vel, double max_acc, double startV, double endV){
    double critical_len; //the critical length of two-order optimal control, acc is the input
    if(startV>max_vel||endV>max_vel){
      std::cerr << "start or end vel is larger than max_vel" << std::endl;
    }
    double startv2 = pow(startV,2);
    double endv2 = pow(endV,2);
    double maxv2 = pow(max_vel,2);
    critical_len = (maxv2-startv2)/(2*max_acc)+(maxv2-endv2)/(2*max_acc);
    if(locallength>=critical_len){
      double t1 = (max_vel-startV)/max_acc;
      double t2 = t1+(locallength-critical_len)/max_vel;
      if(curt<=t1){
        return startV*curt + 0.5*max_acc*pow(curt,2);
      }
      else if(curt<=t2){
        return startV*t1 + 0.5*max_acc*pow(t1,2)+(curt-t1)*max_vel;
      }
      else{
        return startV*t1 + 0.5*max_acc*pow(t1,2) + (t2-t1)*max_vel + max_vel*(curt-t2)-0.5*max_acc*pow(curt-t2,2);
      }
    }
    else{
      double tmpv = sqrt(0.5*(startv2+endv2+2*max_acc*locallength));
      double tmpt = (tmpv-startV)/max_acc;
      if(curt<=tmpt){
        return startV*curt+0.5*max_acc*pow(curt,2);
      }
      else{
        return startV*tmpt+0.5*max_acc*pow(tmpt,2) + tmpv*(curt-tmpt)-0.5*max_acc*pow(curt-tmpt,2);
      }
    }
  }

  std::vector<PathNodePtr> KinoAstar::getVisitedNodes()
  {
    vector<PathNodePtr> visited;
    visited.assign(path_node_pool_.begin(), path_node_pool_.begin() + use_node_num_ - 1);
    return visited;
  }

  Eigen::Vector2i KinoAstar::posToIndex(Eigen::Vector2d pt)
  {
    int idx = std::round((pt[0]-x_lower_)/resolution_);
    int idy = std::round((pt[1]-y_lower_)/resolution_);
    return Eigen::Vector2i(idx,idy);
  }

  int KinoAstar::yawToIndex(double yaw)
  {
    yaw = normalize_angle(yaw);
    if(yaw>3.15||yaw<-3.15){
      std::cerr << "yaw angle is out of range!" << std::endl;
    }
    int idx = floor((yaw - yaw_origin_) * inv_yaw_resolution_);
    return idx;
  }
  void KinoAstar::getFlatState(Eigen::Vector4d state, Eigen::Vector2d control_input,
                                  Eigen::MatrixXd &flat_state, int singul)
  {

    flat_state.resize(2, 3);
    double angle = state(2);
    double vel   = state(3); // vel > 0 

    Eigen::Matrix2d init_R;
    init_R << cos(angle),  -sin(angle),
              sin(angle),   cos(angle);
    if (abs(vel) <= non_siguav){
      vel = singul * non_siguav;
    }
    else{
      vel = singul * vel;
    }
    flat_state << state.head(2), init_R*Eigen::Vector2d(vel, 0.0), 
                  init_R*Eigen::Vector2d(control_input(1), std::tan(control_input(0)) / wheel_base * std::pow(vel, 2));

  }

} // namespace resilient_planner
