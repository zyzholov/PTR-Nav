#include <plan_manage/plan_manage.h>
// #include <tf2/tf2.h>  // Not needed in ROS2
#include <tools/tic_toc.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <thread>
#include <utility>
using namespace plan_manage;
using namespace std;

namespace {

double wrapToPiLocal(double ang)
{
    while (ang > M_PI) ang -= 2.0 * M_PI;
    while (ang < -M_PI) ang += 2.0 * M_PI;
    return ang;
}

double cross2d(const Eigen::Vector2d &a, const Eigen::Vector2d &b)
{
    return a.x() * b.y() - a.y() * b.x();
}

bool onSegment2d(const Eigen::Vector2d &a,
                 const Eigen::Vector2d &b,
                 const Eigen::Vector2d &p)
{
    constexpr double eps = 1.0e-6;
    return std::min(a.x(), b.x()) - eps <= p.x() && p.x() <= std::max(a.x(), b.x()) + eps &&
           std::min(a.y(), b.y()) - eps <= p.y() && p.y() <= std::max(a.y(), b.y()) + eps &&
           std::fabs(cross2d(b - a, p - a)) <= eps;
}

bool segmentsIntersect2d(const Eigen::Vector2d &a,
                         const Eigen::Vector2d &b,
                         const Eigen::Vector2d &c,
                         const Eigen::Vector2d &d)
{
    constexpr double eps = 1.0e-6;
    if (std::max(a.x(), b.x()) + eps < std::min(c.x(), d.x()) ||
        std::max(c.x(), d.x()) + eps < std::min(a.x(), b.x()) ||
        std::max(a.y(), b.y()) + eps < std::min(c.y(), d.y()) ||
        std::max(c.y(), d.y()) + eps < std::min(a.y(), b.y()))
    {
        return false;
    }

    const double o1 = cross2d(b - a, c - a);
    const double o2 = cross2d(b - a, d - a);
    const double o3 = cross2d(d - c, a - c);
    const double o4 = cross2d(d - c, b - c);

    if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps)))
    {
        return true;
    }

    return (std::fabs(o1) <= eps && onSegment2d(a, b, c)) ||
           (std::fabs(o2) <= eps && onSegment2d(a, b, d)) ||
           (std::fabs(o3) <= eps && onSegment2d(c, d, a)) ||
           (std::fabs(o4) <= eps && onSegment2d(c, d, b));
}

bool validatePolylineLocalGeometry(const std::vector<Eigen::Vector2d> &raw_pts,
                                   std::string &reason)
{
    reason.clear();

    std::vector<Eigen::Vector2d> pts;
    pts.reserve(raw_pts.size());
    for (const auto &p : raw_pts)
    {
        if (!p.allFinite())
        {
            reason = "non-finite point";
            return false;
        }
        if (pts.empty() || (p - pts.back()).norm() > 0.035)
            pts.push_back(p);
    }

    if (pts.size() < 4)
        return true;

    std::vector<double> arc(pts.size(), 0.0);
    std::vector<double> seg_len(pts.size() - 1, 0.0);
    std::vector<double> seg_yaw(pts.size() - 1, 0.0);
    for (size_t i = 0; i + 1 < pts.size(); ++i)
    {
        const Eigen::Vector2d d = pts[i + 1] - pts[i];
        const double len = d.norm();
        seg_len[i] = len;
        seg_yaw[i] = (len > 1.0e-6) ? std::atan2(d.y(), d.x()) : 0.0;
        arc[i + 1] = arc[i] + len;
    }

    if (arc.back() < 0.70)
        return true;

    int prev_valid_seg = -1;
    for (size_t i = 0; i < seg_len.size(); ++i)
    {
        if (seg_len[i] < 0.08)
            continue;

        if (prev_valid_seg >= 0)
        {
            const double turn = std::fabs(wrapToPiLocal(seg_yaw[i] - seg_yaw[prev_valid_seg]));
            const double local_arc = seg_len[i] + seg_len[prev_valid_seg];
            const double local_curv = turn / std::max(0.05, 0.5 * local_arc);

            if (turn > 2.15 && local_arc < 1.25)
            {
                std::ostringstream oss;
                oss << "cusp turn " << turn << "rad over " << local_arc << "m";
                reason = oss.str();
                return false;
            }
            if (turn > 1.45 && local_curv > 2.80)
            {
                std::ostringstream oss;
                oss << "local kink curvature " << local_curv << "rad/m, turn=" << turn;
                reason = oss.str();
                return false;
            }
        }

        prev_valid_seg = static_cast<int>(i);
    }

    for (size_t i = 0; i + 4 < pts.size(); ++i)
    {
        for (size_t j = i + 4; j < pts.size(); ++j)
        {
            const double arc_sep = arc[j] - arc[i];
            if (arc_sep < 1.20)
                continue;
            if (arc_sep > 4.50)
                break;

            const double chord = (pts[j] - pts[i]).norm();
            if (arc_sep > 1.20 && chord < 0.32)
            {
                std::ostringstream oss;
                oss << "local loop returns within " << chord << "m after " << arc_sep << "m";
                reason = oss.str();
                return false;
            }
            if (arc_sep > 1.70 && chord / arc_sep < 0.24)
            {
                std::ostringstream oss;
                oss << "folded window chord/arc=" << (chord / arc_sep);
                reason = oss.str();
                return false;
            }
        }
    }

    for (size_t i = 0; i + 1 < pts.size(); ++i)
    {
        if (seg_len[i] < 0.05)
            continue;
        for (size_t j = i + 2; j + 1 < pts.size(); ++j)
        {
            if (j == i + 1 || seg_len[j] < 0.05)
                continue;
            if (segmentsIntersect2d(pts[i], pts[i + 1], pts[j], pts[j + 1]))
            {
                std::ostringstream oss;
                oss << "polyline self-intersection between segments " << i << " and " << j;
                reason = oss.str();
                return false;
            }
        }
    }

    return true;
}

} // namespace

void PlanManager::init(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<se2_grid::SE2Grid> se2_grid_map, std::shared_ptr<se2_grid::SE2Grid> sdf_grid_map){
    node_ = node;
    config_.reset(new Config(node_));
    RCLCPP_INFO(node_->get_logger(), "Planner config: max_iter=%d aml=%d search_time=%.2f pieceTime=%.2f",
                config_->max_iterations, config_->amlMaxIter, config_->max_seach_time, config_->pieceTime);
    se2_grid_map_ = se2_grid_map;
    sdf_grid_map_ = sdf_grid_map;
    hasTarget = false;
    hasOdom = false;
    
    // 初始化状态
    start_state_ = Eigen::Vector4d::Zero();
    end_state_ = Eigen::Vector4d::Zero();
    start_time_ = 0.0;
    
    cudaWarmTimer = node_->create_wall_timer(
        std::chrono::milliseconds(10), std::bind(&PlanManager::warm, this));

    pieceTime = config_->pieceTime;
    kino_path_finder_.reset(new path_searching::KinoAstar);
    kino_path_finder_->init(config_, node_);  
    if (node_->has_parameter("front_end/allow_reverse"))
        front_end_allow_reverse_ = node_->get_parameter("front_end/allow_reverse").as_bool();
    else
        front_end_allow_reverse_ = node_->declare_parameter<bool>("front_end/allow_reverse", false);
    
    // 初始化地图前验证 — SDF map is now optional (planner uses terrain risk)
    RCLCPP_INFO(node_->get_logger(), "Initializing maps...");
    if(!sdf_grid_map_){
        RCLCPP_WARN(node_->get_logger(), "sdf_grid_map is null (optional) — will use terrain risk for collision checking");
    }
    if(!se2_grid_map_){
        RCLCPP_ERROR(node_->get_logger(), "se2_grid_map is null!");
        return;
    }
    
    // Layer data may arrive later from terrain_analyzer callbacks.
    // Do not fail init here; runtime checks in FSM/process will guard planning.
    RCLCPP_INFO(node_->get_logger(), "SE2Grid layers will be validated at planning time");
    
    // Primary: use risk map for collision checking. SDF is optional fallback.
    // Always set the risk/frontend map so map range and resolution are available.
    // The risk layer may arrive later — query_risk will default to free until then.
    if (se2_grid_map_)
        kino_path_finder_->intialRiskMap(se2_grid_map_);
    if (sdf_grid_map_)
        kino_path_finder_->intialMap(sdf_grid_map_);

    Eigen::Vector2d map_XY{config_->mapX, config_->mapY};
    trajCmdPub = node_->create_publisher<mpc_controller::msg::DPtrajContainer>("/planner/dptrajectory", 1);
    auto path_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    visPathPub = node_->create_publisher<nav_msgs::msg::Path>("/visualization/refinedTraj_kinoastar", path_qos);
    publish_optimized_path_visualization_ =
        node_->declare_parameter<bool>("publish_optimized_path_visualization", true);
    if (publish_optimized_path_visualization_)
    {
        visPathKeepaliveTimer = node_->create_wall_timer(
            std::chrono::milliseconds(2000), std::bind(&PlanManager::republishVisualizationPath, this));
    }
    // Fixed-rate trajectory republish is disabled by default — the MPC slides
    // ref_tracking_time_ forward internally on every getCmd() call, so external
    // re-publishing with updated start_time is unnecessary and causes MPC to
    // rebuild xref / revalidate on every publish, which creates micro-stutters.
    traj_republish_hz_ = node_->declare_parameter<double>("traj_republish_hz", 0.0);
    if (traj_republish_hz_ > 0.5)
    {
        traj_republish_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / std::max(1.0, traj_republish_hz_))),
            std::bind(&PlanManager::republishTrajectory, this));
    }
    publish_rough_preview_ = node_->declare_parameter<bool>("publish_rough_preview", false);
    publish_local_horizon_ = node_->declare_parameter<bool>("publish_local_horizon", true);
    local_publish_horizon_sec_ = node_->declare_parameter<double>("local_publish_horizon_sec", 10.0);
    partial_frontier_goal_gap_thresh_ = node_->declare_parameter<double>("partial_frontier_goal_gap_thresh", 1.6);
    partial_frontier_max_length_ = node_->declare_parameter<double>("partial_frontier_max_length", 12.0);
    max_piece_num_per_seg_ = node_->declare_parameter<int>("max_piece_num_per_seg", 8);
    max_piece_num_per_seg_partial_ = node_->declare_parameter<int>("max_piece_num_per_seg_partial", 5);
    rough_path_smoothing_passes_ = node_->declare_parameter<int>("rough_path_smoothing/passes", 2);
    rough_path_smoothing_alpha_ = node_->declare_parameter<double>("rough_path_smoothing/alpha", 0.32);
    rough_path_smoothing_max_deviation_ =
        node_->declare_parameter<double>("rough_path_smoothing/max_deviation", 0.28);
    rough_path_smoothing_risk_slack_ =
        node_->declare_parameter<double>("rough_path_smoothing/risk_slack", 0.03);
    rough_path_smoothing_passes_ = std::clamp(rough_path_smoothing_passes_, 0, 5);
    rough_path_smoothing_alpha_ = std::clamp(rough_path_smoothing_alpha_, 0.0, 0.60);
    rough_path_smoothing_max_deviation_ = std::clamp(rough_path_smoothing_max_deviation_, 0.05, 0.60);
    rough_path_smoothing_risk_slack_ = std::clamp(rough_path_smoothing_risk_slack_, 0.0, 0.20);

    rt_max_iterations_cap_ = node_->declare_parameter<int>("rt/max_iterations_cap", 450);
    rt_aml_max_iter_cap_ = node_->declare_parameter<int>("rt/aml_max_iter_cap", 10);
    config_->max_iterations = std::max(20, std::min(config_->max_iterations, rt_max_iterations_cap_));
    config_->amlMaxIter = std::max(1, std::min(config_->amlMaxIter, rt_aml_max_iter_cap_));
    traj_publish_min_sdf_ = node_->declare_parameter<double>(
        "traj_publish_min_sdf", std::max(0.04, config_->safeMargin * 0.22));
    traj_publish_sdf_tolerance_ = node_->declare_parameter<double>("traj_publish_sdf_tolerance", 0.004);
    traj_publish_risk_hard_ = node_->declare_parameter<double>("traj_publish_risk_hard", 0.84);
    traj_publish_require_valid_layer_ = node_->declare_parameter<bool>("traj_publish_require_valid_layer", false);
    traj_publish_max_start_gap_ = node_->declare_parameter<double>("traj_publish_max_start_gap", 0.35);
    traj_publish_max_start_yaw_gap_ = node_->declare_parameter<double>("traj_publish_max_start_yaw_gap", 0.75);
    traj_safety_ignore_start_sec_ = node_->declare_parameter<double>("traj_safety_ignore_start_sec", 0.60);
    traj_safety_ignore_start_dist_ = node_->declare_parameter<double>("traj_safety_ignore_start_dist", 0.35);
    traj_safety_check_horizon_sec_ = node_->declare_parameter<double>("traj_safety_check_horizon_sec", 6.0);
    traj_publish_max_duration_ = node_->declare_parameter<double>("traj_publish_max_duration", 4.0);
    traj_publish_allow_partial_on_gate_ = node_->declare_parameter<bool>("traj_publish_allow_partial_on_gate", true);
    traj_publish_partial_min_safe_sec_ = node_->declare_parameter<double>("traj_publish_partial_min_safe_sec", 1.0);
    traj_publish_partial_backoff_sec_ = node_->declare_parameter<double>("traj_publish_partial_backoff_sec", 0.35);
    RCLCPP_INFO(node_->get_logger(),
                "Trajectory safety gate: min_sdf=%.3f (tol=%.3f), hard_risk=%.2f, require_valid=%s, start_gate=(%.2fm, %.2frad), ignore_start=(%.2fs, %.2fm), horizon=%.2fs, partial=(%s, min=%.2fs, backoff=%.2fs), rough_preview=%s",
                traj_publish_min_sdf_, traj_publish_sdf_tolerance_, traj_publish_risk_hard_,
                traj_publish_require_valid_layer_ ? "true" : "false",
                traj_publish_max_start_gap_, traj_publish_max_start_yaw_gap_,
                traj_safety_ignore_start_sec_, traj_safety_ignore_start_dist_,
                traj_safety_check_horizon_sec_,
                traj_publish_allow_partial_on_gate_ ? "on" : "off",
                traj_publish_partial_min_safe_sec_, traj_publish_partial_backoff_sec_,
                publish_rough_preview_ ? "on" : "off");
    RCLCPP_INFO(node_->get_logger(),
                "Visualization horizon: local_only=%s horizon=%.2fs",
                publish_local_horizon_ ? "on" : "off", local_publish_horizon_sec_);
    RCLCPP_INFO(node_->get_logger(),
                "Optimized path visualization: %s",
                publish_optimized_path_visualization_ ? "on" : "off");
    RCLCPP_INFO(node_->get_logger(),
                "Realtime caps: max_iterations=%d amlMaxIter=%d, frontier_gap=%.2fm frontier_len=%.2fm, piece_cap=(%d/%d)",
                config_->max_iterations, config_->amlMaxIter,
                partial_frontier_goal_gap_thresh_, partial_frontier_max_length_,
                max_piece_num_per_seg_, max_piece_num_per_seg_partial_);
    RCLCPP_INFO(node_->get_logger(),
                "Rough path smoothing: passes=%d alpha=%.2f max_dev=%.2fm risk_slack=%.2f",
                rough_path_smoothing_passes_, rough_path_smoothing_alpha_,
                rough_path_smoothing_max_deviation_, rough_path_smoothing_risk_slack_);
    clearVisualizationPath();
    
    RCLCPP_INFO(node_->get_logger(), "PlanManager initialized successfully");
}
void PlanManager::warm(){
    return;
}

bool PlanManager::process(){
    if(!hasTarget || !hasOdom){
        RCLCPP_DEBUG(node_->get_logger(), "Waiting for target or odom: target=%d, odom=%d", hasTarget, hasOdom);
        return false;
    } 
    
    RCLCPP_DEBUG(node_->get_logger(), "Triggering trajectory planning");
    hasTarget = false;
    begin_time = node_->get_clock()->now();
    
    vector<Eigen::Vector3d> sampleTraj;
    path_searching::KinoTrajData kino_trajs_;
    std::vector<Eigen::Vector3d> visKinoPath;

    kino_path_finder_->reset();
    Eigen::Vector4d iniFs, finFs;
    iniFs = start_state_;
    finFs = end_state_;

    auto buildDirectTerminalSeed = [&](const Eigen::Vector4d &start,
                                       Eigen::Vector4d &goal,
                                       std::vector<Eigen::Vector3d> &seed) -> bool {
        seed.clear();
        const Eigen::Vector2d delta = goal.head(2) - start.head(2);
        const double dist = delta.norm();
        if (dist < 0.08 || dist > 2.4)
            return false;

        const double line_yaw = std::atan2(delta.y(), delta.x());
        double target_yaw = goal(2);
        const double start_gap = std::fabs(wrapToPiLocal(line_yaw - start(2)));
        const double end_gap = std::fabs(wrapToPiLocal(line_yaw - target_yaw));
        const double yaw_span = std::fabs(wrapToPiLocal(target_yaw - start(2)));
        const double implied_cur = yaw_span / std::max(0.35, dist);
        if (end_gap > 1.25 || implied_cur > std::max(2.20, config_->kmax * 3.0))
        {
            const Eigen::Vector2d goal_dir(std::cos(target_yaw), std::sin(target_yaw));
            const double approach_len = std::clamp(std::min(1.20, dist * 0.45), 0.45, 1.20);
            if (dist < approach_len + 0.25)
                return false;

            const Eigen::Vector2d pre_goal = goal.head(2) - approach_len * goal_dir;
            const Eigen::Vector2d pre_delta = pre_goal - start.head(2);
            const double pre_dist = pre_delta.norm();
            if (pre_dist < 0.25)
                return false;

            const double pre_line_yaw = std::atan2(pre_delta.y(), pre_delta.x());
            const double pre_start_gap = std::fabs(wrapToPiLocal(pre_line_yaw - start(2)));
            const double pre_end_gap = std::fabs(wrapToPiLocal(pre_line_yaw - target_yaw));
            if (pre_start_gap > 1.30 || pre_end_gap > 1.45)
                return false;

            const int steps1 = std::max(3, static_cast<int>(std::ceil(pre_dist / 0.16)));
            const double dyaw1 = wrapToPiLocal(target_yaw - start(2));
            seed.reserve(steps1 + 8);
            for (int i = 0; i <= steps1; ++i)
            {
                const double r = static_cast<double>(i) / static_cast<double>(steps1);
                const double s = r * r * (3.0 - 2.0 * r);
                Eigen::Vector3d p;
                p.head(2) = start.head(2) + r * pre_delta;
                p(2) = wrapToPiLocal(start(2) + s * dyaw1);
                if (kino_path_finder_->isCollided(p))
                {
                    seed.clear();
                    return false;
                }
                seed.push_back(p);
            }

            const int steps2 = std::max(2, static_cast<int>(std::ceil(approach_len / 0.16)));
            for (int i = 1; i <= steps2; ++i)
            {
                const double r = static_cast<double>(i) / static_cast<double>(steps2);
                Eigen::Vector3d p;
                p.head(2) = pre_goal + r * (goal.head(2) - pre_goal);
                p(2) = target_yaw;
                if (kino_path_finder_->isCollided(p))
                {
                    seed.clear();
                    return false;
                }
                seed.push_back(p);
            }
            return seed.size() >= 3;
        }

        const bool reverse_seed = start_gap > 1.75 && dist < 1.5;
        if (!reverse_seed && start_gap > 1.20)
            return false;

        const int steps = std::max(3, static_cast<int>(std::ceil(dist / 0.16)));
        const double dyaw = wrapToPiLocal(target_yaw - start(2));
        seed.reserve(steps + 1);
        for (int i = 0; i <= steps; ++i)
        {
            const double r = static_cast<double>(i) / static_cast<double>(steps);
            Eigen::Vector3d p;
            p.head(2) = start.head(2) + r * delta;
            p(2) = wrapToPiLocal(start(2) + r * dyaw);
            if (kino_path_finder_->isCollided(p))
            {
                seed.clear();
                return false;
            }
            seed.push_back(p);
        }
        return seed.size() >= 3;
    };

    double model_time = 0.0;
    bool using_direct_terminal_seed = false;
    int status = kino_path_finder_->search(iniFs, Eigen::Vector2d::Zero(), finFs, model_time, true);
    if(status == 0) {
        if (buildDirectTerminalSeed(iniFs, finFs, sampleTraj))
        {
            using_direct_terminal_seed = true;
            RCLCPP_WARN(node_->get_logger(),
                        "Kino-A* failed near goal; use checked direct terminal seed with %zu samples",
                        sampleTraj.size());
        }
        else
        {
            RCLCPP_ERROR(node_->get_logger(), "No path found by kino_astar!");
            return false;
        }
    }
    if (!using_direct_terminal_seed)
    {
        std::vector<Eigen::Vector3d> ts = kino_path_finder_->getRoughSamples();
        if(ts.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "getRoughSamples returned empty trajectory!");
            return false;
        }
        sampleTraj.insert(sampleTraj.end(), ts.begin(), ts.end());
    }

    auto trimByLength = [](const std::vector<Eigen::Vector3d> &in, double max_len) {
        if (in.size() <= 2 || max_len <= 0.0)
            return in;

        std::vector<Eigen::Vector3d> out;
        out.reserve(in.size());
        out.push_back(in.front());

        double acc_len = 0.0;
        for (size_t i = 1; i < in.size(); ++i)
        {
            const Eigen::Vector2d prev = in[i - 1].head(2);
            const Eigen::Vector2d cur = in[i].head(2);
            const Eigen::Vector2d delta = cur - prev;
            const double seg_len = delta.norm();
            if (seg_len < 1.0e-5)
                continue;

            const double seg_yaw = std::atan2(delta.y(), delta.x());
            if (acc_len + seg_len >= max_len)
            {
                const double r = std::clamp((max_len - acc_len) / seg_len, 0.0, 1.0);
                Eigen::Vector3d cut;
                cut.head(2) = prev + r * delta;
                cut(2) = seg_yaw;
                if ((cut.head(2) - out.back().head(2)).norm() > 0.04)
                    out.push_back(cut);
                else
                    out.back() = cut;
                break;
            }

            Eigen::Vector3d p = in[i];
            p(2) = seg_yaw;
            out.push_back(p);
            acc_len += seg_len;
        }

        if (out.size() == 1 && in.size() > 1)
            out.push_back(in[1]);
        return out;
    };

	    auto pathLength = [](const std::vector<Eigen::Vector3d> &path) {
	        double len = 0.0;
	        for (size_t i = 1; i < path.size(); ++i)
	            len += (path[i].head(2) - path[i - 1].head(2)).norm();
	        return len;
	    };

    auto appendTerminalConnector = [&](std::vector<Eigen::Vector3d> &path,
                                       const Eigen::Vector4d &goal,
                                       double max_gap) -> bool {
        if (path.empty())
            return false;

        const Eigen::Vector3d last = path.back();
        const Eigen::Vector2d delta = goal.head(2) - last.head(2);
        const double gap = delta.norm();
        if (gap < 0.08)
        {
            path.back().head(2) = goal.head(2);
            path.back()(2) = goal(2);
            return true;
        }
        if (gap > max_gap)
            return false;

        const double line_yaw = std::atan2(delta.y(), delta.x());
        const double start_gap = std::fabs(wrapToPiLocal(line_yaw - last(2)));
        const double end_gap = std::fabs(wrapToPiLocal(line_yaw - goal(2)));
        const double yaw_span = std::fabs(wrapToPiLocal(goal(2) - last(2)));
        const double implied_cur = yaw_span / std::max(0.35, gap);
        if (start_gap > 0.85 || end_gap > 0.85 || implied_cur > std::max(1.50, config_->kmax * 2.4))
            return false;

        const int steps = std::max(2, static_cast<int>(std::ceil(gap / 0.18)));
        const double dyaw = wrapToPiLocal(goal(2) - last(2));
        std::vector<Eigen::Vector3d> connector;
        connector.reserve(steps);
        for (int i = 1; i <= steps; ++i)
        {
            const double r = static_cast<double>(i) / static_cast<double>(steps);
            Eigen::Vector3d p;
            p.head(2) = last.head(2) + r * delta;
            p(2) = wrapToPiLocal(last(2) + r * dyaw);
            if (kino_path_finder_->isCollided(p))
                return false;
            connector.push_back(p);
        }

        path.insert(path.end(), connector.begin(), connector.end());
        return true;
    };

    auto appendCheckedGoalConnector = [&](std::vector<Eigen::Vector3d> &path,
                                          Eigen::Vector4d &goal,
                                          double max_gap) -> bool {
        if (path.empty())
            return false;

        struct ConnectorCandidate
        {
            size_t anchor_idx = 0;
            Eigen::Vector4d goal;
            std::vector<Eigen::Vector3d> connector;
            double score = std::numeric_limits<double>::infinity();
        };

        const size_t original_size = path.size();
        const size_t max_scan = std::min<size_t>(original_size, 90);
        ConnectorCandidate best;
        bool found = false;

        auto connector_geometry_ok = [&](size_t anchor_idx,
                                         const std::vector<Eigen::Vector3d> &connector,
                                         std::string &reason) -> bool {
            std::vector<Eigen::Vector2d> local_path;
            const size_t begin_idx = (anchor_idx > 7) ? (anchor_idx - 7) : 0;
            local_path.reserve(anchor_idx - begin_idx + 1 + connector.size());
            for (size_t i = begin_idx; i <= anchor_idx && i < path.size(); ++i)
                local_path.push_back(path[i].head(2));
            for (const auto &p : connector)
                local_path.push_back(p.head(2));
            return validatePolylineLocalGeometry(local_path, reason);
        };

        auto build_candidate = [&](size_t anchor_idx, ConnectorCandidate &cand) -> bool {
            const Eigen::Vector3d anchor = path[anchor_idx];
            const Eigen::Vector2d delta = goal.head(2) - anchor.head(2);
            const double gap = delta.norm();
            if (gap > max_gap)
                return false;

            cand.anchor_idx = anchor_idx;
            cand.goal = goal;
            cand.connector.clear();

            if (gap < 0.08)
            {
                cand.goal.head(2) = goal.head(2);
                cand.score = 0.05 * static_cast<double>(original_size - 1 - anchor_idx);
                return true;
            }

            double line_yaw = std::atan2(delta.y(), delta.x());
            double start_gap = std::fabs(wrapToPiLocal(line_yaw - anchor(2)));

            double target_yaw = cand.goal(2);
            const double end_gap = std::fabs(wrapToPiLocal(line_yaw - target_yaw));
            const double yaw_span = std::fabs(wrapToPiLocal(target_yaw - anchor(2)));
            const double implied_cur = yaw_span / std::max(0.35, gap);
            if (end_gap > 0.90 || implied_cur > std::max(1.25, config_->kmax * 2.0))
            {
                const Eigen::Vector2d goal_dir(std::cos(target_yaw), std::sin(target_yaw));
                const double approach_len = std::clamp(std::min(2.0, gap * 0.45), 0.75, 2.0);
                if (gap < approach_len + 0.35)
                    return false;

                const Eigen::Vector2d pre_goal = goal.head(2) - approach_len * goal_dir;
                const Eigen::Vector2d pre_delta = pre_goal - anchor.head(2);
                const double pre_gap = pre_delta.norm();
                if (pre_gap < 0.25 || pre_gap > max_gap)
                    return false;

                line_yaw = std::atan2(pre_delta.y(), pre_delta.x());
                start_gap = std::fabs(wrapToPiLocal(line_yaw - anchor(2)));
                const double pre_end_gap = std::fabs(wrapToPiLocal(line_yaw - target_yaw));
                if (start_gap > 1.05 || pre_end_gap > 1.20)
                    return false;

                const int steps1 = std::max(3, static_cast<int>(std::ceil(pre_gap / 0.25)));
                const double dyaw1 = wrapToPiLocal(target_yaw - anchor(2));
                cand.connector.reserve(steps1 + 4);
                for (int i = 1; i <= steps1; ++i)
                {
                    const double r = static_cast<double>(i) / static_cast<double>(steps1);
                    const double s = r * r * (3.0 - 2.0 * r);
                    Eigen::Vector3d p;
                    p.head(2) = anchor.head(2) + r * pre_delta;
                    p(2) = wrapToPiLocal(anchor(2) + s * dyaw1);
                    if (kino_path_finder_->isCollided(p))
                        return false;
                    cand.connector.push_back(p);
                }

                const int steps2 = std::max(2, static_cast<int>(std::ceil(approach_len / 0.25)));
                for (int i = 1; i <= steps2; ++i)
                {
                    const double r = static_cast<double>(i) / static_cast<double>(steps2);
                    Eigen::Vector3d p;
                    p.head(2) = pre_goal + r * (goal.head(2) - pre_goal);
                    p(2) = target_yaw;
                    if (kino_path_finder_->isCollided(p))
                        return false;
                    cand.connector.push_back(p);
                }

                std::string geom_reason;
                if (!connector_geometry_ok(anchor_idx, cand.connector, geom_reason))
                    return false;

                const double truncated = static_cast<double>(original_size - 1 - anchor_idx);
                cand.score = pre_gap + approach_len + 2.80 * start_gap + 0.60 * pre_end_gap + 0.08 * truncated;
                return true;
            }
            if (start_gap > 0.95)
                return false;

            const int steps = std::max(3, static_cast<int>(std::ceil(gap / 0.25)));
            const double dyaw = wrapToPiLocal(target_yaw - anchor(2));
            cand.connector.reserve(steps);
            for (int i = 1; i <= steps; ++i)
            {
                const double r = static_cast<double>(i) / static_cast<double>(steps);
                const double s = r * r * (3.0 - 2.0 * r);
                Eigen::Vector3d p;
                p.head(2) = anchor.head(2) + r * delta;
                p(2) = wrapToPiLocal(anchor(2) + s * dyaw);
                if (kino_path_finder_->isCollided(p))
                    return false;
                cand.connector.push_back(p);
            }

            std::string geom_reason;
            if (!connector_geometry_ok(anchor_idx, cand.connector, geom_reason))
                return false;

            const double truncated = static_cast<double>(original_size - 1 - anchor_idx);
            cand.score = gap + 2.80 * start_gap + 0.08 * truncated;
            return true;
        };

        for (size_t off = 0; off < max_scan; off += 3)
        {
            const size_t idx = original_size - 1 - off;
            ConnectorCandidate cand;
            if (build_candidate(idx, cand) && cand.score < best.score)
            {
                best = std::move(cand);
                found = true;
            }
        }

        if (!found)
            return false;

        const size_t removed = original_size - 1 - best.anchor_idx;
        if (best.anchor_idx + 1 < path.size())
            path.erase(path.begin() + static_cast<long>(best.anchor_idx + 1), path.end());
        if (best.connector.empty())
        {
            path.back().head(2) = best.goal.head(2);
            path.back()(2) = best.goal(2);
        }
        else
        {
            path.insert(path.end(), best.connector.begin(), best.connector.end());
        }
        goal = best.goal;

        if (removed > 0)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Goal connector replaced %zu rough tail samples before reaching final target",
                        removed);
        }
        return true;
    };

    Eigen::Vector4d opt_finFs = finFs;
    bool using_partial_frontier = false;
    if (!sampleTraj.empty())
    {
        const Eigen::Vector2d rough_end_xy = sampleTraj.back().head(2);
        const double rough_goal_gap = (rough_end_xy - finFs.head(2)).norm();
        const double start_goal_dist = (iniFs.head(2) - finFs.head(2)).norm();
        
        if (rough_goal_gap > 0.80)
        {
            if (start_goal_dist < 2.2)
            {
                Eigen::Vector4d connector_goal = opt_finFs;
                bool terminal_connected = appendCheckedGoalConnector(sampleTraj, connector_goal, 2.20);

                if (terminal_connected)
                {
                    opt_finFs = connector_goal;
                    RCLCPP_WARN(node_->get_logger(),
                                "Front-end rough path ended %.2fm before final goal; appended checked terminal connector for backend optimization",
                                rough_goal_gap);
                }
                else
                {
                    RCLCPP_WARN(node_->get_logger(),
                                "Front-end stopped %.2fm before final goal during final approach (start_goal=%.2fm); reject instead of forcing a kinked backend closure",
                                rough_goal_gap, start_goal_dist);
                    return false;
                }
            }
            else
            {
                Eigen::Vector4d connector_goal = finFs;
                const double connect_limit = std::min(25.0, std::max(4.0, partial_frontier_max_length_));
	                if (appendCheckedGoalConnector(sampleTraj, connector_goal, connect_limit))
	                {
	                    const double rolling_horizon_len =
	                        std::clamp(std::max(7.5, partial_frontier_max_length_), 7.5, 12.0);
	                    if (start_goal_dist > rolling_horizon_len + 5.0)
	                    {
	                        sampleTraj = trimByLength(sampleTraj, rolling_horizon_len);
	                        const Eigen::Vector3d &local_end = sampleTraj.back();
	                        opt_finFs.head(3) = local_end;
	                        using_partial_frontier = true;
	                        RCLCPP_WARN(node_->get_logger(),
	                                    "Front-end timed out %.2fm before goal; optimize rolling horizon %.2fm to local endpoint [%.2f, %.2f, %.2f]",
	                                    rough_goal_gap, pathLength(sampleTraj),
	                                    opt_finFs(0), opt_finFs(1), opt_finFs(2));
	                    }
	                    else
	                    {
	                        opt_finFs = connector_goal;
	                        RCLCPP_WARN(node_->get_logger(),
	                                    "Front-end timed out %.2fm before goal, but checked connector reached final target; optimize full goal path",
	                                    rough_goal_gap);
	                    }
	                }
	                else
	                {
	                    using_partial_frontier = true;
	                    if (rough_goal_gap > partial_frontier_goal_gap_thresh_) {
	                        sampleTraj = trimByLength(sampleTraj, std::max(3.0, partial_frontier_max_length_));
	                    }
	                    const double partial_len = pathLength(sampleTraj);
	                    if (start_goal_dist > 8.0 && partial_len < 4.0)
	                    {
	                        RCLCPP_WARN(node_->get_logger(),
	                                    "Reject short partial frontier %.2fm for distant goal %.2fm; keep previous trajectory and retry",
	                                    partial_len, start_goal_dist);
	                        return false;
	                    }
	                    const Eigen::Vector3d &local_end = sampleTraj.back();
	                    opt_finFs.head(3) = local_end;
                    RCLCPP_WARN(node_->get_logger(),
                                "Front-end partial path (gap=%.2fm), switch to local endpoint [%.2f, %.2f, %.2f]",
                                rough_goal_gap, opt_finFs(0), opt_finFs(1), opt_finFs(2));
                }
            }
        }
    }

    auto sanitizeSampleTraj = [&]() -> bool {
        if (sampleTraj.empty())
            return false;

        std::vector<Eigen::Vector3d> cleaned;
        cleaned.reserve(sampleTraj.size());
        for (auto p : sampleTraj)
        {
            if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
                continue;
            p(2) = wrapToPiLocal(p(2));
            if (!cleaned.empty())
            {
                const double dxy = (p.head(2) - cleaned.back().head(2)).norm();
                const double dyaw = std::fabs(wrapToPiLocal(p(2) - cleaned.back()(2)));
                if (dxy < 0.04 && dyaw < 0.05)
                    continue;
                if (dxy < 1.0e-4)
                {
                    cleaned.back()(2) = p(2);
                    continue;
                }
            }
            cleaned.push_back(p);
        }

        if (cleaned.size() < 2)
            return false;
        sampleTraj.swap(cleaned);
        return true;
    };

    if (!sanitizeSampleTraj())
    {
        RCLCPP_ERROR(node_->get_logger(), "Rough sample trajectory is degenerate after sanitizing");
        return false;
    }

    auto sampleRiskForSmoothing = [&](const Eigen::Vector2d &xy) -> double {
        if (!se2_grid_map_ || !se2_grid_map_->exists("risk"))
            return std::numeric_limits<double>::quiet_NaN();
        try {
            const double risk = se2_grid_map_->getValue("risk", Eigen::Vector3d(xy.x(), xy.y(), 0.0));
            return std::isfinite(risk) ? risk : std::numeric_limits<double>::quiet_NaN();
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };

    auto segmentIsFreeForSmoothing = [&](const Eigen::Vector3d &a,
                                         const Eigen::Vector3d &b) -> bool {
        const Eigen::Vector2d delta = b.head(2) - a.head(2);
        const double len = delta.norm();
        const int steps = std::max(1, static_cast<int>(std::ceil(len / 0.12)));
        const double dyaw = wrapToPiLocal(b(2) - a(2));
        for (int k = 0; k <= steps; ++k)
        {
            const double r = static_cast<double>(k) / static_cast<double>(steps);
            Eigen::Vector3d q;
            q.head(2) = a.head(2) + r * delta;
            q(2) = wrapToPiLocal(a(2) + r * dyaw);
            if (kino_path_finder_->isCollided(q))
                return false;
        }
        return true;
    };

    auto refreshInteriorYawFromGeometry = [&]() {
        if (sampleTraj.size() < 3)
            return;
        for (size_t i = 1; i + 1 < sampleTraj.size(); ++i)
        {
            const Eigen::Vector2d tangent = sampleTraj[i + 1].head(2) - sampleTraj[i - 1].head(2);
            if (tangent.norm() > 1.0e-4)
                sampleTraj[i](2) = std::atan2(tangent.y(), tangent.x());
        }
    };

    auto smoothSampleTraj = [&]() {
        if (rough_path_smoothing_passes_ <= 0 ||
            rough_path_smoothing_alpha_ <= 1.0e-4 ||
            sampleTraj.size() < 5)
        {
            return;
        }

        const std::vector<Eigen::Vector3d> original = sampleTraj;
        int accepted_moves = 0;

        refreshInteriorYawFromGeometry();
        for (int pass = 0; pass < rough_path_smoothing_passes_; ++pass)
        {
            std::vector<Eigen::Vector3d> next = sampleTraj;
            for (size_t i = 1; i + 1 < sampleTraj.size(); ++i)
            {
                const Eigen::Vector2d average =
                    0.5 * (sampleTraj[i - 1].head(2) + sampleTraj[i + 1].head(2));
                Eigen::Vector2d xy =
                    sampleTraj[i].head(2) +
                    rough_path_smoothing_alpha_ * (average - sampleTraj[i].head(2));

                Eigen::Vector2d deviation = xy - original[i].head(2);
                const double dev_norm = deviation.norm();
                if (dev_norm > rough_path_smoothing_max_deviation_)
                {
                    xy = original[i].head(2) +
                         deviation * (rough_path_smoothing_max_deviation_ / dev_norm);
                }

                Eigen::Vector3d cand = sampleTraj[i];
                cand.head(2) = xy;

                if (kino_path_finder_->isCollided(cand))
                    continue;

                const double old_risk = sampleRiskForSmoothing(sampleTraj[i].head(2));
                const double new_risk = sampleRiskForSmoothing(cand.head(2));
                if (std::isfinite(old_risk) && std::isfinite(new_risk) &&
                    new_risk > old_risk + rough_path_smoothing_risk_slack_)
                {
                    continue;
                }

                if (!segmentIsFreeForSmoothing(next[i - 1], cand) ||
                    !segmentIsFreeForSmoothing(cand, sampleTraj[i + 1]))
                {
                    continue;
                }

                next[i] = cand;
                ++accepted_moves;
            }
            sampleTraj.swap(next);
            refreshInteriorYawFromGeometry();
        }

        if (accepted_moves > 0)
        {
            RCLCPP_DEBUG(node_->get_logger(),
                         "Smoothed rough path: accepted %d point moves over %d passes",
                         accepted_moves, rough_path_smoothing_passes_);
        }
    };

    smoothSampleTraj();

    auto enforceForwardSampleConsistency = [&]() -> bool {
        if (sampleTraj.size() < 2)
            return false;

        sampleTraj.front()(2) = iniFs(2);
        for (size_t i = 1; i + 1 < sampleTraj.size(); ++i)
        {
            const Eigen::Vector2d tangent = sampleTraj[i + 1].head(2) - sampleTraj[i - 1].head(2);
            if (tangent.norm() > 1.0e-4)
                sampleTraj[i](2) = std::atan2(tangent.y(), tangent.x());
        }

        const bool exact_final_goal =
            !using_partial_frontier && (opt_finFs.head(2) - finFs.head(2)).norm() < 0.15;
        if (exact_final_goal)
        {
            sampleTraj.back()(2) = opt_finFs(2);
        }
        else
        {
            const Eigen::Vector2d tail_tangent =
                sampleTraj.back().head(2) - sampleTraj[sampleTraj.size() - 2].head(2);
            if (tail_tangent.norm() > 1.0e-4)
                sampleTraj.back()(2) = std::atan2(tail_tangent.y(), tail_tangent.x());
        }

        if (!front_end_allow_reverse_)
        {
            double reverse_like_len = 0.0;
            for (size_t i = 0; i + 1 < sampleTraj.size(); ++i)
            {
                const Eigen::Vector2d step = sampleTraj[i + 1].head(2) - sampleTraj[i].head(2);
                const double len = step.norm();
                if (len < 1.0e-4)
                    continue;
                const Eigen::Vector2d heading(std::cos(sampleTraj[i](2)), std::sin(sampleTraj[i](2)));
                if (step.dot(heading) < -0.03)
                    reverse_like_len += len;
            }
            if (reverse_like_len > 0.25)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Reject rough path before backend: %.2fm reverse-like motion while reverse is disabled",
                            reverse_like_len);
                return false;
            }
        }

        const Eigen::Vector2d mission = finFs.head(2) - iniFs.head(2);
        const double mission_len = mission.norm();
        if (mission_len > 2.5)
        {
            const Eigen::Vector2d dir = mission / mission_len;
            double max_progress = -std::numeric_limits<double>::infinity();
            size_t max_idx = 0;
            for (size_t i = 0; i < sampleTraj.size(); ++i)
            {
                const double progress = (sampleTraj[i].head(2) - iniFs.head(2)).dot(dir);
                if (progress > max_progress)
                {
                    max_progress = progress;
                    max_idx = i;
                }
            }
            const double last_progress = (sampleTraj.back().head(2) - iniFs.head(2)).dot(dir);
            const double regression_limit = std::clamp(0.22 * mission_len, 0.90, 2.00);
            if (max_idx + 2 < sampleTraj.size() &&
                max_progress - last_progress > regression_limit)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Reject rough path before backend: tail regresses %.2fm along mission direction (limit %.2fm)",
                            max_progress - last_progress, regression_limit);
                return false;
            }
        }

        std::vector<Eigen::Vector2d> rough_xy;
        rough_xy.reserve(sampleTraj.size());
        for (const auto &p : sampleTraj)
            rough_xy.push_back(p.head(2));
        std::string geom_reason;
        if (!validatePolylineLocalGeometry(rough_xy, geom_reason))
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Reject rough path before backend: %s",
                        geom_reason.c_str());
            return false;
        }

        return true;
    };

    if (!enforceForwardSampleConsistency())
        return false;

    // Rough preview is optional; enabling it may create visual mismatch with optimized path in RViz.
    if (publish_rough_preview_)
    {
        nav_msgs::msg::Path rough_msg;
        rough_msg.header.frame_id = "map";
        rough_msg.header.stamp = node_->get_clock()->now();
        for (const auto &p : sampleTraj)
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = rough_msg.header;
            pose.pose.position.x = p.x();
            pose.pose.position.y = p.y();
            double z_vis = 0.05;
            try {
                const double elev = se2_grid_map_->getValue("elevation", Eigen::Vector3d(p.x(), p.y(), 0.0));
                const double var = se2_grid_map_->getValue("var", Eigen::Vector3d(p.x(), p.y(), 0.0));
                if (std::isfinite(elev) && std::isfinite(var) && var < 0.30)
                {
                    z_vis = elev + 0.03;
                }
            } catch (...) {
                z_vis = 0.05;
            }
            pose.pose.position.z = z_vis;
            pose.pose.orientation.w = 1.0;
            rough_msg.poses.push_back(pose);
        }
        if (visPathPub != nullptr && !rough_msg.poses.empty())
        {
            visPathPub->publish(rough_msg);
            std::lock_guard<std::mutex> viz_lock(viz_mutex_);
            lastVisPath = rough_msg;
            hasVisPath = true;
        }
    }
    
    RCLCPP_DEBUG(node_->get_logger(), "Getting kino nodes...");
    kino_path_finder_->getKinoNode(kino_trajs_, sampleTraj);
    RCLCPP_DEBUG(node_->get_logger(), "Got %lu kino nodes", kino_trajs_.size());
    
    for(int i = 0; i < sampleTraj.size(); i++){
        Eigen::Vector3d pos;
        pos.head(2) = sampleTraj[i].head(2);
        pos[2] = 0.2;
        visKinoPath.push_back(pos);
    }

    int segnum = kino_trajs_.size();
    if(segnum == 0){
        RCLCPP_ERROR(node_->get_logger(), "getKinoNode returned empty trajectory! Path search may have failed.");
        return false;
    }
    
    RCLCPP_DEBUG(node_->get_logger(), "Kino trajectory segments: %d", segnum);
    
    std::vector<int> refined_singuals;
    refined_singuals.resize(segnum);
    
    Eigen::VectorXd refined_rt;
    refined_rt.resize(segnum);
    
    std::vector<Eigen::MatrixXd> refined_inPs_container;
    refined_inPs_container.resize(segnum);
    
    std::vector<Eigen::Vector2d> refined_gearPos;
    std::vector<double> refined_angles; 
    refined_gearPos.resize(segnum - 1);
    refined_angles.resize(segnum - 1);
    
    double basetime = 0.0;
    std::vector<int> pnums;
    
    for(int i = 0; i < segnum; i++){
        double timePerPiece = pieceTime;
        
        // 边界检查
        if(i >= kino_trajs_.size()){
            RCLCPP_ERROR(node_->get_logger(), "Index %d out of range for kino_trajs_ (size: %lu)", i, kino_trajs_.size());
            return false;
        }
        
        path_searching::FlatTrajData kino_traj = kino_trajs_[i];
        refined_singuals[i] = kino_traj.singul;
        std::vector<Eigen::Vector3d> pts = kino_traj.traj_pts;
        
        if(pts.empty()){
            RCLCPP_WARN(node_->get_logger(), "Trajectory segment %d has no points", i);
            refined_rt[i] = timePerPiece;
            refined_inPs_container[i].resize(2, 0);
            continue;
        }
        
        int piece_nums;
        double initTotalduration = 0.0;
        for(const auto &pt : pts){
            if (!pt.allFinite() || !std::isfinite(pt[2]) || pt[2] <= 1.0e-5)
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "Invalid kino segment point before optimization: seg=%d pt=(%.3f, %.3f, %.6f)",
                             i, pt[0], pt[1], pt[2]);
                return false;
            }
            initTotalduration += pt[2];
        }
        if (!std::isfinite(initTotalduration) || initTotalduration <= 1.0e-5)
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "Invalid kino segment duration before optimization: seg=%d duration=%.6f",
                         i, initTotalduration);
            return false;
        }
        
        RCLCPP_DEBUG(node_->get_logger(), "Segment %d: initTotalduration=%.4f, pieceTime=%.4f", i, initTotalduration, timePerPiece);
        
        piece_nums = std::max(int(initTotalduration / timePerPiece + 0.5), 1);
        const int max_piece_nums = std::max(
            2, using_partial_frontier ? max_piece_num_per_seg_partial_ : max_piece_num_per_seg_);
        if (piece_nums > max_piece_nums)
        {
            RCLCPP_DEBUG(node_->get_logger(),
                        "Segment %d piece_nums capped from %d to %d (duration=%.2f, pieceTime=%.2f)",
                        i, piece_nums, max_piece_nums, initTotalduration, timePerPiece);
            piece_nums = max_piece_nums;
        }
        double dt = initTotalduration / piece_nums;
        if (!std::isfinite(dt) || dt <= 1.0e-5)
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "Invalid refined segment dt before optimization: seg=%d dt=%.6f",
                         i, dt);
            return false;
        }
        refined_rt[i] = (initTotalduration / piece_nums);
        pnums.push_back(piece_nums);
        
        RCLCPP_DEBUG(node_->get_logger(), "Segment %d: piece_nums=%d", i, piece_nums);

        refined_inPs_container[i].resize(2, piece_nums - 1);
        for(int j = 0; j < piece_nums - 1; j++){
            double t = basetime + (j+1)*dt;
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(t);
            if (!pos.allFinite())
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "evaluatePos returned invalid point: seg=%d inner=%d t=%.3f",
                             i, j, t);
                return false;
            }
            refined_inPs_container[i].col(j) = pos.head(2);
        }
        
        if(i >= 1 && i-1 < refined_gearPos.size()){
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(basetime);
            if (!pos.allFinite())
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "evaluatePos returned invalid gear point: seg=%d t=%.3f",
                             i, basetime);
                return false;
            }
            refined_gearPos[i-1] = pos.head(2);
            refined_angles[i-1] = pos[2];
        }
        
        basetime += initTotalduration;
    }
    
    // Start velocity: use the vehicle's actual speed from the start state.
    // Use a very small floor only to prevent numerical degeneracy in the
    // optimizer when the vehicle is completely stopped.
    double ini_vel = std::abs(iniFs(3));
    // This velocity is the spatial MINCO boundary tangent, not the commanded
    // vehicle speed. Keep a small floor so the optimized path has a well-defined
    // start heading even when the vehicle is nearly stopped.
    if (ini_vel < 0.10) ini_vel = 0.10;

    // End velocity: 0 at the exact goal (vehicle should stop). For partial
    // paths, maintain a non-zero velocity to prevent the optimizer from
    // forcing a rigid stop at an arbitrary intermediate point (which causes
    // twisted terminal segments).
    const double dist_to_final_goal = (end_state_.head(2) - opt_finFs.head(2)).norm();
    const double local_dist = (opt_finFs.head(2) - iniFs.head(2)).norm();
    double end_vel_mag = 0.0;
    if (using_partial_frontier || dist_to_final_goal > 0.10) {
        // Smoothly decay velocity with distance to final goal.
        // At 5m from goal: ~2.0m/s, at 1m: ~0.6m/s, at 0.3m: ~0.2m/s.
        double speed_by_dist = std::min(config_->vmax, std::max(0.20, dist_to_final_goal * 2.0));
        end_vel_mag = std::min(speed_by_dist, std::max(0.30, local_dist * 1.2));
    } else {
        // At the final goal the temporal profile still stops because t(s) has
        // zero end derivative. The spatial path, however, must keep a small
        // terminal tangent along the requested goal yaw; otherwise getYaw(T)
        // becomes ill-defined and the publish gate may reject a valid trajectory.
        end_vel_mag = 0.10;
    }

    iniState2d << iniFs[0], refined_singuals[0] * ini_vel * cos(iniFs[2]), 0.0,
                iniFs[1], refined_singuals[0] * ini_vel * sin(iniFs[2]), 0.0;
    finState2d << opt_finFs[0], refined_singuals[segnum-1] * end_vel_mag * cos(opt_finFs[2]), 0.0,
                opt_finFs[1], refined_singuals[segnum-1] * end_vel_mag * sin(opt_finFs[2]), 0.0;

    if (!iniState2d.allFinite() || !finState2d.allFinite())
    {
        RCLCPP_ERROR(node_->get_logger(), "Invalid boundary state before OptimizeSe2Trajectory");
        return false;
    }
    for (int i = 0; i < refined_rt.size(); ++i)
    {
        if (!std::isfinite(refined_rt[i]) || refined_rt[i] <= 1.0e-5 ||
            !refined_inPs_container[i].allFinite())
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "Invalid optimization input: seg=%d dt=%.6f inPs=(%ldx%ld)",
                         i, refined_rt[i],
                         refined_inPs_container[i].rows(), refined_inPs_container[i].cols());
            return false;
        }
    }
    for (size_t i = 0; i < refined_gearPos.size(); ++i)
    {
        if (!refined_gearPos[i].allFinite() || !std::isfinite(refined_angles[i]))
        {
            RCLCPP_ERROR(node_->get_logger(), "Invalid gear constraint before OptimizeSe2Trajectory: idx=%zu", i);
            return false;
        }
    }
    
    // 验证参数的合理性
    RCLCPP_DEBUG(node_->get_logger(), "Validation before OptimizeSe2Trajectory:");
    RCLCPP_DEBUG(node_->get_logger(), "  segnum=%d, gearPos.size=%lu, angles.size=%lu",
                segnum, refined_gearPos.size(), refined_angles.size());
    for(int i = 0; i < refined_inPs_container.size(); i++){
        RCLCPP_DEBUG(node_->get_logger(), "  inPs_container[%d]: rows=%ld, cols=%ld",
                    i, refined_inPs_container[i].rows(), refined_inPs_container[i].cols());
    }
    RCLCPP_DEBUG(node_->get_logger(), "  refined_rt: size=%ld, values=[%.3f, %.3f, ...]",
                refined_rt.size(), refined_rt.size() > 0 ? refined_rt[0] : 0.0,
                refined_rt.size() > 1 ? refined_rt[1] : 0.0);
    
    PolyTrajOpt::TrajOpt refinedOpt;
    int flagSs = 0;
    try {
        RCLCPP_DEBUG(node_->get_logger(), "Calling OptimizeSe2Trajectory...");
        flagSs = refinedOpt.OptimizeSe2Trajectory(
            iniState2d, finState2d, refined_rt,
            refined_inPs_container, refined_gearPos,
            refined_angles, se2_grid_map_, nullptr, config_, refined_singuals, vis_tool, "kinoastar");
        RCLCPP_DEBUG(node_->get_logger(), "OptimizeSe2Trajectory returned: %d", flagSs);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Exception in OptimizeSe2Trajectory: %s", e.what());
        return false;
    } catch (...) {
        RCLCPP_ERROR(node_->get_logger(), "Unknown exception in OptimizeSe2Trajectory");
        return false;
    }
    
    if(!flagSs){
        RCLCPP_ERROR(node_->get_logger(), "OptimizeSe2Trajectory failed with return value: %d", flagSs);
        return false;
    }
    
    double max_cur = refinedOpt.getMaxCur();
    (void)max_cur;
    PolyTrajOpt::UgvTrajectory candidate_traj = refinedOpt.getOptTraj();
    candidate_traj.start_time = start_time_;

    // 验证 optTraj 的有效性
    if(candidate_traj.Traj_container.empty() || candidate_traj.etas.empty()){
        RCLCPP_ERROR(node_->get_logger(), "OptTraj is empty! Traj_container size: %lu, etas size: %lu", 
                     candidate_traj.Traj_container.size(), candidate_traj.etas.size());
        return false;
    }

    auto validateCandidateGeometry = [&](const PolyTrajOpt::UgvTrajectory &traj) -> bool {
        if (!front_end_allow_reverse_)
        {
            for (const int eta : traj.etas)
            {
                if (eta <= 0)
                {
                    RCLCPP_WARN(node_->get_logger(),
                                "Reject optimized trajectory before publish: reverse segment generated while reverse is disabled");
                    return false;
                }
            }
        }

        const double total_dur = traj.getTotalDuration();
        if (!std::isfinite(total_dur) || total_dur <= 0.0)
            return false;

        const Eigen::Vector2d mission = finFs.head(2) - iniFs.head(2);
        const double mission_len = mission.norm();
        const bool check_mission_progress = mission_len > 2.5;
        Eigen::Vector2d dir = Eigen::Vector2d::Zero();
        if (check_mission_progress)
            dir = mission / mission_len;
        const double dt = std::max(0.08, std::min(0.20, total_dur / 45.0));
        std::vector<Eigen::Vector2d> sampled_xy;
        sampled_xy.reserve(static_cast<size_t>(std::ceil(total_dur / dt)) + 2);
        double max_progress = -std::numeric_limits<double>::infinity();
        double last_progress = 0.0;
        double reverse_like_len = 0.0;
        bool have_prev = false;
        Eigen::Vector2d prev_pos = Eigen::Vector2d::Zero();
        double prev_yaw = iniFs(2);
        int sample_idx = 0;
        int max_idx = 0;

        for (double t = 0.0; t <= total_dur + 1e-6; t += dt, ++sample_idx)
        {
            Eigen::Vector2d pos;
            double yaw = prev_yaw;
            try {
                pos = traj.getPos(std::min(t, total_dur));
                yaw = traj.getYaw(std::min(t, total_dur));
            } catch (...) {
                return false;
            }

            if (!pos.allFinite() || !std::isfinite(yaw))
                return false;

            sampled_xy.push_back(pos);

            if (check_mission_progress)
            {
                const double progress = (pos - iniFs.head(2)).dot(dir);
                last_progress = progress;
                if (progress > max_progress)
                {
                    max_progress = progress;
                    max_idx = sample_idx;
                }
            }

            if (have_prev && !front_end_allow_reverse_)
            {
                const Eigen::Vector2d step = pos - prev_pos;
                const double len = step.norm();
                if (len > 1.0e-4)
                {
                    const Eigen::Vector2d heading(std::cos(prev_yaw), std::sin(prev_yaw));
                    if (step.dot(heading) < -0.03)
                        reverse_like_len += len;
                }
            }

            prev_pos = pos;
            prev_yaw = yaw;
            have_prev = true;
        }

        if (!front_end_allow_reverse_ && reverse_like_len > 0.30)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Reject optimized trajectory before publish: %.2fm reverse-like motion while reverse is disabled",
                            reverse_like_len);
            return false;
        }

        if (check_mission_progress)
        {
            const double regression_limit = std::clamp(0.22 * mission_len, 0.90, 2.00);
            if (max_idx + 2 < sample_idx && max_progress - last_progress > regression_limit)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Reject optimized trajectory before publish: tail regresses %.2fm along mission direction (limit %.2fm)",
                            max_progress - last_progress, regression_limit);
                return false;
            }
        }

        std::string geom_reason;
        if (!validatePolylineLocalGeometry(sampled_xy, geom_reason))
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Reject optimized trajectory before publish: %s",
                        geom_reason.c_str());
            return false;
        }

        return true;
    };

    if (!validateCandidateGeometry(candidate_traj))
        return false;

    try {
        const Eigen::Vector2d traj_start = candidate_traj.getPos(0.0);
        const double traj_start_yaw = candidate_traj.getYaw(0.0);
        const double start_gap = (traj_start - start_state_.head(2)).norm();
        const double start_yaw_gap = std::fabs(wrapToPiLocal(traj_start_yaw - start_state_(2)));
        if (start_gap > std::max(0.05, traj_publish_max_start_gap_) ||
            start_yaw_gap > std::max(0.15, traj_publish_max_start_yaw_gap_))
        {
            RCLCPP_ERROR(node_->get_logger(),
                         "Reject optimized trajectory before publish: start discontinuity gap=%.2fm>%.2fm yaw_gap=%.2frad>%.2frad (traj=[%.2f, %.2f, %.2f], start=[%.2f, %.2f, %.2f])",
                         start_gap, traj_publish_max_start_gap_,
                         start_yaw_gap, traj_publish_max_start_yaw_gap_,
                         traj_start.x(), traj_start.y(), traj_start_yaw,
                         start_state_(0), start_state_(1), start_state_(2));
            return false;
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Reject optimized trajectory before publish: cannot sample start state: %s",
                     e.what());
        return false;
    }

    if (!using_partial_frontier)
    {
        try {
            const double total_dur = candidate_traj.getTotalDuration();
            const Eigen::Vector2d traj_tail = candidate_traj.getPos(total_dur);
            const double yaw_sample_backoff = std::min(0.12, std::max(0.03, total_dur * 0.02));
            const double traj_tail_yaw = candidate_traj.getYaw(std::max(0.0, total_dur - yaw_sample_backoff));
            const double tail_gap = (traj_tail - opt_finFs.head(2)).norm();
            if (tail_gap > 0.22)
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "Reject optimized trajectory before publish: terminal gap %.2fm to goal [%.2f, %.2f]",
                             tail_gap, opt_finFs(0), opt_finFs(1));
                return false;
            }
            const double tail_yaw_gap = std::fabs(wrapToPiLocal(traj_tail_yaw - opt_finFs(2)));
            if (tail_yaw_gap > 0.28)
            {
                RCLCPP_ERROR(node_->get_logger(),
                             "Reject optimized trajectory before publish: terminal yaw gap %.2frad to goal yaw %.2f (sample %.2fs before tail)",
                             tail_yaw_gap, opt_finFs(2), yaw_sample_backoff);
                return false;
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(node_->get_logger(),
                         "Reject optimized trajectory before publish: cannot sample terminal state: %s",
                         e.what());
            return false;
        }
    }

    double min_sdf = std::numeric_limits<double>::infinity();
    double max_risk = -std::numeric_limits<double>::infinity();
    if (!validateTrajectorySafety(candidate_traj, min_sdf, max_risk)) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Reject optimized trajectory before publish: min_sdf=%.3f, max_risk=%.3f",
                     min_sdf, max_risk);
        return false;
    }
    RCLCPP_DEBUG(node_->get_logger(), "Trajectory safety gate passed: min_sdf=%.3f, max_risk=%.3f",
                 min_sdf, max_risk);
    
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        optTraj = candidate_traj;
        published_traj_duration_ = optTraj.getTotalDuration();
    }

    // ego_planner-style: sample optimized trajectory for next replan reuse.
    // Commit this guidance only after all publish gates pass; rejected
    // candidates should not bias the next front-end search.
    try {
        const double total_dur = candidate_traj.getTotalDuration();
        prev_samples_.clear();
        for (double t = 0.0; t < total_dur; t += 0.25)
        {
            Eigen::Vector2d pos = candidate_traj.getPos(t);
            double yaw = candidate_traj.getYaw(t);
            prev_samples_.push_back(Eigen::Vector3d(pos.x(), pos.y(), yaw));
        }
        prev_end_state_ = end_state_;
        has_prev_samples_ = true;
        Eigen::Vector2d d = candidate_traj.getPos(std::min(total_dur, 4.0)) - candidate_traj.getPos(0.0);
        kino_path_finder_->setPrevTrajDirection(d);
    } catch (...) {}

    RCLCPP_DEBUG(node_->get_logger(), "dptraj total arc: %f", candidate_traj.getTotalArc());
    RCLCPP_DEBUG(node_->get_logger(), "dptraj traj time: %f", candidate_traj.getTotalDuration());
    RCLCPP_DEBUG(node_->get_logger(), "dptraj segment number: %d", candidate_traj.getSegNum());

    // Publish visualization path — disabled in the live Gazebo profile because
    // the MPC publishes a much lighter sliding reference marker.
    if (!publish_optimized_path_visualization_)
        return true;

    // Clear old one first via empty message
    {
        nav_msgs::msg::Path clear_msg;
        clear_msg.header.frame_id = "map";
        clear_msg.header.stamp = node_->get_clock()->now();
        if (visPathPub != nullptr) visPathPub->publish(clear_msg);

        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = node_->get_clock()->now();
        
        double total_duration = candidate_traj.getTotalDuration();
        if(total_duration <= 0){
            RCLCPP_WARN(node_->get_logger(), "Total trajectory duration is invalid: %f", total_duration);
            return false;
        }
        
        double vis_duration = total_duration;
        if (publish_local_horizon_)
        {
            vis_duration = std::min(total_duration, std::max(2.0, local_publish_horizon_sec_));
        }
        else
        {
            vis_duration = std::min(total_duration, std::max(8.0, traj_safety_check_horizon_sec_ + 1.0));
        }
        
        // Strictly match RViz path duration with what MPC will actually receive
        if (traj_publish_partial_active_ && traj_publish_partial_duration_ > 0.0) {
            vis_duration = std::min(vis_duration, traj_publish_partial_duration_);
        }

        Eigen::Vector2d min_range = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
        Eigen::Vector2d max_range = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
        bool has_map_range = false;
        if (sdf_grid_map_)
        {
            sdf_grid_map_->getMapRange(min_range, max_range);
            has_map_range = true;
        }

        double step = std::max(0.05, vis_duration / 100.0);
        double last_valid_z = 0.05;

        // Offset visualization start to account for planning latency.
        // The trajectory starts from where the vehicle WAS at replan time.
        // By the time it's published, the vehicle has moved forward (typically
        // 0.2-0.5s at 2m/s).  Starting from t=0 shows the "past" portion and
        // makes the reference look mismatched in RViz.
        double vis_start_t = 0.0;
        {
            double planning_elapsed = node_->get_clock()->now().seconds() - start_time_;
            vis_start_t = std::max(0.0, std::min(planning_elapsed - 0.08, total_duration * 0.35));
        }

        for (double t = vis_start_t; t <= vis_duration + 1e-6; t += step) {
            try {
                Eigen::Vector2d pt = candidate_traj.getPos(t);

                // Try to get elevation for better z-display, but never skip a point
                // — gaps in the displayed path look like planning failures to the user.
                double z_vis = last_valid_z;
                try {
                    const double elev = se2_grid_map_->getValue("elevation", Eigen::Vector3d(pt[0], pt[1], 0.0));
                    const double var = se2_grid_map_->getValue("var", Eigen::Vector3d(pt[0], pt[1], 0.0));
                    if (std::isfinite(elev) && std::isfinite(var) && var < 0.35)
                    {
                        z_vis = elev + 0.03;
                        last_valid_z = z_vis;
                    }
                } catch (...) {}

                geometry_msgs::msg::PoseStamped pose;
                pose.header = path_msg.header;
                pose.pose.position.x = pt[0];
                pose.pose.position.y = pt[1];
                pose.pose.position.z = z_vis;
                pose.pose.orientation.w = 1.0;
                path_msg.poses.push_back(pose);
            } catch (const std::exception& e) {
                RCLCPP_WARN(node_->get_logger(), "Error sampling trajectory at t=%f: %s", t, e.what());
                break;
            }
        }
        
        if (visPathPub != nullptr && !path_msg.poses.empty()) {
            visPathPub->publish(path_msg);
            std::lock_guard<std::mutex> viz_lock(viz_mutex_);
            lastVisPath = path_msg;
            hasVisPath = true;
            RCLCPP_DEBUG(node_->get_logger(),
                        "Published visualization path with %lu poses (duration=%.2fs)",
                        path_msg.poses.size(), vis_duration);
        } else {
            RCLCPP_WARN(node_->get_logger(), "No poses generated for visualization");
        }
    }

    return true;
}

bool PlanManager::replanLightweight()
{
    if (optTraj.getSegNum() <= 0 || !hasTarget || !hasOdom)
        return false;

    const double total_dur = optTraj.getTotalDuration();
    if (total_dur <= 0.4)  // relaxed from 0.8 — allow shorter trajectories
        return false;

    // Find where the vehicle is on the current trajectory, so we sample
    // from the vehicle's actual progress point rather than a fixed 40%.
    // This preserves the front portion that the vehicle is executing and
    // only re-optimizes the back portion — exactly like ego_planner.
    const Eigen::Vector2d veh_pos = start_state_.head(2);
    double nearest_t = 0.0;
    double nearest_dist = std::numeric_limits<double>::max();
    for (double t = 0.0; t < total_dur; t += 0.10)
    {
        try {
            Eigen::Vector2d pt = optTraj.getPos(t);
            double d = (pt - veh_pos).squaredNorm();
            if (d < nearest_dist) { nearest_dist = d; nearest_t = t; }
        } catch (...) { break; }
    }
    // Clamp: only look forward, never backward.  total_dur-1.5 would force
    // looking at a point behind the vehicle for short trajectories.
    nearest_t = std::max(0.05, std::min(nearest_t, total_dur - 0.4));
    if (nearest_t >= total_dur - 0.2)
        return false;

    const double sample_start_t = nearest_t;
    if (sample_start_t >= total_dur - 0.2)
        return false;

    // Sample tail of current optimized trajectory from nearest_t onward.
    // The optimizer's iniState2d hard-constrains the start to the vehicle
    // position, so the first sample point should be slightly ahead on the
    // old trajectory to guide the shape, not at the vehicle position itself.
    std::vector<Eigen::Vector3d> samples;
    const double sample_dt = 0.20;
    for (double t = sample_start_t; t < total_dur - 0.05; t += sample_dt)
    {
        try {
            Eigen::Vector2d pos = optTraj.getPos(t);
            double yaw = optTraj.getYaw(t);
            samples.push_back(Eigen::Vector3d(pos.x(), pos.y(), yaw));
        } catch (...) { break; }
    }

    if (samples.size() < 3)
        return false;

    // Extend toward goal
    const Eigen::Vector3d &last_pt = samples.back();
    const Eigen::Vector2d goal_xy = end_state_.head(2);
    const Eigen::Vector2d delta = goal_xy - last_pt.head(2);
    const double dist_to_goal = delta.norm();
    if (dist_to_goal < 0.3)
        return false;  // already at goal

    const Eigen::Vector2d dir_to_goal = delta / dist_to_goal;
    double yaw_to_goal = std::atan2(dir_to_goal.y(), dir_to_goal.x());
    double yaw_diff = std::abs(wrapToPiLocal(yaw_to_goal - last_pt(2)));
    if (yaw_diff > 0.6) {
        RCLCPP_WARN(node_->get_logger(), "Lightweight: yaw diff too large (%.2f), skip", yaw_diff);
        return false;
    }

    const int ext_steps = std::max(2, std::min(30, static_cast<int>(dist_to_goal / 0.6)));
    const double ext_step_len = dist_to_goal / ext_steps;

    // Collision-check extension points
    for (int i = 1; i <= ext_steps; ++i)
    {
        Eigen::Vector2d ext_pt = last_pt.head(2) + dir_to_goal * ext_step_len * i;
        Eigen::Vector3d check_pt(ext_pt.x(), ext_pt.y(), end_state_(2));
        if (kino_path_finder_->isCollided(check_pt))
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: ext collides at step %d/%d → fallback full Kino-A*", i, ext_steps);
            return false;
        }
        samples.push_back(Eigen::Vector3d(ext_pt.x(), ext_pt.y(), end_state_(2)));
    }
    samples.push_back(Eigen::Vector3d(goal_xy.x(), goal_xy.y(), end_state_(2)));

    RCLCPP_INFO(node_->get_logger(),
                "Lightweight: %lu samples (tail=%lu ext=%d), skip Kino-A*",
                samples.size(), samples.size() - ext_steps - 1, ext_steps);

    {
        std::vector<Eigen::Vector2d> sample_xy;
        sample_xy.reserve(samples.size());
        for (const auto &p : samples)
            sample_xy.push_back(p.head(2));
        std::string geom_reason;
        if (!validatePolylineLocalGeometry(sample_xy, geom_reason))
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: reject sampled tail/extension geometry: %s",
                        geom_reason.c_str());
            return false;
        }
    }

    // Convert samples to segments (bypasses Kino-A* search)
    path_searching::KinoTrajData kino_trajs_;
    kino_path_finder_->getKinoNode(kino_trajs_, samples);
    int segnum = kino_trajs_.size();
    if (segnum == 0) return false;

    // Build optimization input
    std::vector<int> refined_singuals(segnum);
    Eigen::VectorXd refined_rt(segnum);
    std::vector<Eigen::MatrixXd> refined_inPs_container(segnum);
    std::vector<Eigen::Vector2d> refined_gearPos(segnum > 1 ? segnum - 1 : 0);
    std::vector<double> refined_angles(segnum > 1 ? segnum - 1 : 0);
    double basetime = 0.0;

    for (int i = 0; i < segnum; i++)
    {
        path_searching::FlatTrajData kino_traj = kino_trajs_[i];
        refined_singuals[i] = kino_traj.singul;
        std::vector<Eigen::Vector3d> pts = kino_traj.traj_pts;
        if (pts.empty()) { refined_rt[i] = pieceTime; refined_inPs_container[i].resize(2, 0); continue; }

        double initTotalduration = 0.0;
        for (const auto &pt : pts)
        {
            if (!pt.allFinite() || !std::isfinite(pt[2]) || pt[2] <= 1.0e-5)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Lightweight: invalid kino segment point seg=%d pt=(%.3f, %.3f, %.6f)",
                            i, pt[0], pt[1], pt[2]);
                return false;
            }
            initTotalduration += pt[2];
        }
        if (!std::isfinite(initTotalduration) || initTotalduration <= 1.0e-5)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: invalid kino segment duration seg=%d duration=%.6f",
                        i, initTotalduration);
            return false;
        }

        int piece_nums = std::max(1, int(initTotalduration / pieceTime + 0.5));
        piece_nums = std::min(piece_nums, 5);
        double dt = initTotalduration / piece_nums;
        if (!std::isfinite(dt) || dt <= 1.0e-5)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: invalid segment dt seg=%d dt=%.6f",
                        i, dt);
            return false;
        }
        refined_rt[i] = dt;

        refined_inPs_container[i].resize(2, piece_nums - 1);
        for (int j = 0; j < piece_nums - 1; j++)
        {
            double t = basetime + (j + 1) * dt;
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(t);
            if (!pos.allFinite())
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Lightweight: evaluatePos invalid seg=%d inner=%d t=%.3f",
                            i, j, t);
                return false;
            }
            refined_inPs_container[i].col(j) = pos.head(2);
        }

        if (i >= 1 && i - 1 < (int)refined_gearPos.size())
        {
            Eigen::Vector3d pos = kino_path_finder_->evaluatePos(basetime);
            if (!pos.allFinite())
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Lightweight: evaluatePos invalid gear seg=%d t=%.3f",
                            i, basetime);
                return false;
            }
            refined_gearPos[i - 1] = pos.head(2);
            refined_angles[i - 1] = pos[2];
        }
        basetime += initTotalduration;
    }

    Eigen::Vector4d iniFs = start_state_;
    Eigen::Vector4d opt_finFs = end_state_;

    Eigen::Matrix<double, 2, 3> iniState2d, finState2d;
    double ini_vel = std::abs(iniFs(3));
    if (ini_vel < 0.05) ini_vel = 0.05;
    // Spatial terminal tangent only; temporal profile still stops at the tail.
    const double end_vel_mag = 0.10;
    iniState2d << iniFs[0], refined_singuals[0] * ini_vel * cos(iniFs[2]), 0.0,
                  iniFs[1], refined_singuals[0] * ini_vel * sin(iniFs[2]), 0.0;
    finState2d << opt_finFs[0], refined_singuals[segnum-1] * end_vel_mag * cos(opt_finFs[2]), 0.0,
                  opt_finFs[1], refined_singuals[segnum-1] * end_vel_mag * sin(opt_finFs[2]), 0.0;

    if (!iniState2d.allFinite() || !finState2d.allFinite())
    {
        RCLCPP_WARN(node_->get_logger(), "Lightweight: invalid boundary state");
        return false;
    }
    for (int i = 0; i < refined_rt.size(); ++i)
    {
        if (!std::isfinite(refined_rt[i]) || refined_rt[i] <= 1.0e-5 ||
            !refined_inPs_container[i].allFinite())
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: invalid optimization input seg=%d dt=%.6f inPs=(%ldx%ld)",
                        i, refined_rt[i],
                        refined_inPs_container[i].rows(), refined_inPs_container[i].cols());
            return false;
        }
    }
    for (size_t i = 0; i < refined_gearPos.size(); ++i)
    {
        if (!refined_gearPos[i].allFinite() || !std::isfinite(refined_angles[i]))
        {
            RCLCPP_WARN(node_->get_logger(), "Lightweight: invalid gear constraint idx=%zu", i);
            return false;
        }
    }

    // Reduced optimization: the warm-start from the current trajectory is
    // already very close to the solution.  Limiting iterations prevents the
    // optimizer from wandering into wild shapes — it only does local
    // "rubber band" smoothing around the current trajectory.
    const int saved_max_iter = config_->max_iterations;
    const int saved_aml = config_->amlMaxIter;
    config_->max_iterations = 25;
    config_->amlMaxIter = 2;

    PolyTrajOpt::TrajOpt refinedOpt;
    int flagSs = 0;
    try {
        flagSs = refinedOpt.OptimizeSe2Trajectory(
            iniState2d, finState2d, refined_rt,
            refined_inPs_container, refined_gearPos,
            refined_angles, se2_grid_map_, nullptr, config_, refined_singuals, vis_tool, "lightweight");
    } catch (...) { flagSs = 0; }

    config_->max_iterations = saved_max_iter;
    config_->amlMaxIter = saved_aml;

    if (!flagSs)
    {
        RCLCPP_WARN(node_->get_logger(), "Lightweight opt failed, fallback");
        return false;
    }

    PolyTrajOpt::UgvTrajectory candidate_traj = refinedOpt.getOptTraj();
    candidate_traj.start_time = start_time_;

    {
        const double cand_dur = candidate_traj.getTotalDuration();
        if (!std::isfinite(cand_dur) || cand_dur <= 0.0)
            return false;
        const double sample_dt = std::max(0.08, std::min(0.20, cand_dur / 45.0));
        std::vector<Eigen::Vector2d> cand_xy;
        cand_xy.reserve(static_cast<size_t>(std::ceil(cand_dur / sample_dt)) + 2);
        for (double t = 0.0; t <= cand_dur + 1.0e-6; t += sample_dt)
        {
            try {
                cand_xy.push_back(candidate_traj.getPos(std::min(t, cand_dur)));
            } catch (...) {
                return false;
            }
        }
        std::string geom_reason;
        if (!validatePolylineLocalGeometry(cand_xy, geom_reason))
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Lightweight: reject optimized geometry: %s",
                        geom_reason.c_str());
            return false;
        }
    }

    double min_sdf = 0, max_risk = 0;
    if (!validateTrajectorySafety(candidate_traj, min_sdf, max_risk))
    {
        RCLCPP_WARN(node_->get_logger(), "Lightweight safety gate fail, fallback");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        optTraj = candidate_traj;
        published_traj_duration_ = optTraj.getTotalDuration();
    }

    try {
        Eigen::Vector2d d = candidate_traj.getPos(std::min(candidate_traj.getTotalDuration(), 4.0)) -
                            candidate_traj.getPos(0.0);
        kino_path_finder_->setPrevTrajDirection(d);
    } catch (...) {}

    RCLCPP_INFO(node_->get_logger(),
                "Lightweight OK: dur=%.2fs seg=%d",
                candidate_traj.getTotalDuration(), candidate_traj.getSegNum());
    return true;
}

bool PlanManager::validateTrajectorySafety(double &min_sdf, double &max_risk)
{
    std::lock_guard<std::mutex> lock(traj_mutex_);
    return validateTrajectorySafety(optTraj, min_sdf, max_risk);
}

bool PlanManager::validateTrajectorySafety(const PolyTrajOpt::UgvTrajectory &traj,
                                           double &min_sdf, double &max_risk)
{
    traj_publish_partial_active_ = false;
    traj_publish_partial_duration_ = -1.0;

    const double total_duration = traj.getTotalDuration();
    if (total_duration <= 0.0) {
        return false;
    }

    Eigen::Vector2d min_range = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Vector2d max_range = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    bool has_map_range = false;
    // Use fused_map (terrain risk) for map range.
    if (se2_grid_map_)
    {
        se2_grid_map_->getMapRange(min_range, max_range);
        has_map_range = true;
    }

    Eigen::Vector2d start_xy(0.0, 0.0);
    bool has_start_xy = false;
    try {
        start_xy = traj.getPos(0.0);
        has_start_xy = true;
    } catch (...) {
    }

    Eigen::Vector2d tail_xy(0.0, 0.0);
    bool final_approach_traj = false;
    try {
        tail_xy = traj.getPos(total_duration);
        final_approach_traj = has_start_xy && ((tail_xy - start_xy).norm() < 2.2);
    } catch (...) {
    }

    bool partial_triggered = false;
    auto try_activate_partial_publish = [&](double hit_t, const char *reason, double value, double threshold) -> bool {
        if (!traj_publish_allow_partial_on_gate_) {
            return false;
        }
        if (final_approach_traj) {
            RCLCPP_WARN(node_->get_logger(),
                        "Safety gate %s at t=%.2f during final approach; reject trajectory instead of publishing a clipped terminal path",
                        reason, hit_t);
            return false;
        }
        const double backoff = std::max(0.05, traj_publish_partial_backoff_sec_);
        const double safe_t = std::min(total_duration, std::max(0.0, hit_t - backoff));
        if (safe_t < std::max(0.20, traj_publish_partial_min_safe_sec_)) {
            return false;
        }

        traj_publish_partial_active_ = true;
        traj_publish_partial_duration_ = safe_t;
        partial_triggered = true;
        RCLCPP_WARN(node_->get_logger(),
                    "Safety gate %s at t=%.2f (value=%.3f thr=%.3f), publish partial trajectory up to %.2fs",
                    reason, hit_t, value, threshold, traj_publish_partial_duration_);
        return true;
    };

    const double step = std::max(0.15, total_duration / 50.0);  // sparser: ~0.15s between checks
    for (double t = 0.0; t <= total_duration + 1e-6; t += step) {
        Eigen::Vector2d pt;
        try {
            pt = traj.getPos(t);
        } catch (...) {
            break;
        }

        const bool short_traj = (total_duration <= traj_safety_ignore_start_sec_ + 1e-3);
        const double moved_dist = has_start_xy ? (pt - start_xy).norm() : traj_safety_ignore_start_dist_;
        const bool within_horizon = (t <= traj_safety_check_horizon_sec_ + 1e-6);
        const bool ignore_start_gate =
            (!short_traj) &&
            ((t < traj_safety_ignore_start_sec_) ||
             (moved_dist < traj_safety_ignore_start_dist_));
        const bool enforce_gate = within_horizon && !ignore_start_gate;

        if (!within_horizon)
            break;

        if (has_map_range &&
            (pt[0] < min_range.x() || pt[0] > max_range.x() || pt[1] < min_range.y() || pt[1] > max_range.y()))
        {
            if (enforce_gate && !try_activate_partial_publish(t, "out_of_map", 1.0, 0.0)) {
                RCLCPP_WARN(node_->get_logger(),
                            "Safety gate hit: t=%.2f pos=(%.2f,%.2f) is out of map range x:[%.2f,%.2f] y:[%.2f,%.2f]",
                            t, pt[0], pt[1], min_range.x(), max_range.x(), min_range.y(), max_range.y());
                return false;
            }
            continue;
        }

        const Eigen::Vector3d query(pt[0], pt[1], 0.0);

        // SDF/ESDF check REMOVED — map is no longer published.  The empty
        // grid shell was triggering hidden try-catch exceptions thousands
        // of times per planning cycle (Gemini: "performance assassin").
        // Collision checking is now exclusively risk-based.

        double risk = std::numeric_limits<double>::quiet_NaN();
        try {
            risk = se2_grid_map_->getValue("risk", query);
        } catch (...) {
        }
        if (std::isfinite(risk)) {
            if (within_horizon)
                max_risk = std::max(max_risk, risk);
            if (enforce_gate && risk >= traj_publish_risk_hard_) {
                if (!try_activate_partial_publish(t, "risk", risk, traj_publish_risk_hard_)) {
                    RCLCPP_WARN(node_->get_logger(),
                                "Safety gate hit: t=%.2f pos=(%.2f,%.2f) risk=%.3f >= %.3f",
                                t, pt[0], pt[1], risk, traj_publish_risk_hard_);
                    return false;
                }
                break;
            }
        }

        // The live terrain valid layer can lag behind sdf/risk. Treat it as a
        // hard publish gate only when explicitly requested.
        double valid = std::numeric_limits<double>::quiet_NaN();
        try {
            valid = se2_grid_map_->getValue("valid", query);
        } catch (...) {
        }
        if (enforce_gate && traj_publish_require_valid_layer_) {
            if (!std::isfinite(valid) || valid < 0.5) {
                if (!try_activate_partial_publish(t, "valid", valid, 0.5)) {
                    RCLCPP_WARN(node_->get_logger(),
                                "Safety gate hit: t=%.2f pos=(%.2f,%.2f) valid=%.3f < 0.50",
                                t, pt[0], pt[1], valid);
                    return false;
                }
                break;
            }
        } else if (enforce_gate && std::isfinite(valid) && valid < 0.5) {
            RCLCPP_WARN_THROTTLE(
                node_->get_logger(), *node_->get_clock(), 1500,
                "Safety gate valid layer is low but ignored: t=%.2f pos=(%.2f,%.2f) valid=%.3f",
                t, pt[0], pt[1], valid);
        }
    }

    // Safety gate: if risk data is unavailable (map not yet populated),
    // accept the trajectory optimistically.  Kino-A* already did collision
    // checking during search.  SDF is no longer published.
    if (!std::isfinite(max_risk)) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                    "Safety gate: no risk data within %.2fs horizon, accepting optimistically",
                    traj_safety_check_horizon_sec_);
    }
    min_sdf = 0.0;  // SDF removed — always report 0
    if (partial_triggered) {
        RCLCPP_WARN(node_->get_logger(),
                    "Trajectory safety gate downgraded to partial publish (duration=%.2fs / total=%.2fs)",
                    traj_publish_partial_duration_, total_duration);
    }
    return true;
}

void PlanManager::publishSimTraj2Controller() {
    std::lock_guard<std::mutex> lock(traj_mutex_);
    if(optTraj.getSegNum() <= 0){
        RCLCPP_WARN(node_->get_logger(), "OptTraj has no segments to publish");
        return;
    }
    
    try {
        const double total_duration = optTraj.getTotalDuration();
        if (total_duration <= 0.0)
        {
            RCLCPP_WARN(node_->get_logger(), "OptTraj duration is invalid: %.3f", total_duration);
            return;
        }

        const double aligned_start_time = optTraj.start_time;

        // Convert start_time (double seconds) to ROS2 Time message
        int sec = static_cast<int>(aligned_start_time);
        uint32_t nsec = static_cast<uint32_t>((aligned_start_time - sec) * 1e9);
        trajmsg.start_time.sec = sec;
        trajmsg.start_time.nanosec = nsec;
        
        trajmsg.traj_container.clear();
        trajmsg.reverse.clear();
        
        // Publish the full trajectory length unless blocked by safety gate.
        // Truncating prevents MPC from driving to the goal if replanning is delayed.
        const double publish_duration =
            (traj_publish_partial_active_ && traj_publish_partial_duration_ > 0.0)
                ? std::min(total_duration, traj_publish_partial_duration_)
                : total_duration;
        double accum_t = 0.0;

        for(int i = 0; i < optTraj.getSegNum(); i++){
            if(i >= optTraj.etas.size()){
                RCLCPP_ERROR(node_->get_logger(), "Index %d out of range for etas", i);
                return;
            }
            
            int singual = optTraj.etas[i];
            mpc_controller::msg::PolyTrajAC trajSegment;
            
            int piece_num = optTraj.Traj_container[i].getPieceNum();
            for(int j = 0; j < piece_num; j++){
                const double piece_dt = optTraj.Traj_container[i].tTraj[j].getDuration();
                                    if (accum_t + piece_dt > publish_duration + 1e-6)
                {
                                            const bool no_piece_selected_yet = trajmsg.traj_container.empty() && trajSegment.trajs.empty();
                                            // In partial mode, keep at least one piece so controller is not starved.
                                            if (traj_publish_partial_active_ && no_piece_selected_yet)
                                            {
                                                // fallthrough and include this first piece
                                            }
                                            else
                                            {
                    break;
                                            }
                }

                mpc_controller::msg::SinglePolyAC piece;
                piece.dt = piece_dt;
                piece.ds = optTraj.Traj_container[i].posTraj[j].getDuration();
                
                Eigen::Matrix<double, 2, 6> c = optTraj.Traj_container[i].posTraj[j].getCoeffMat();
                Eigen::Matrix<double, 1, 6> c1 = optTraj.Traj_container[i].tTraj[j].getCoeffMat(); 
                
                for (int k = 0; k < 6; k++){
                    piece.coef_x.push_back(c(0, k));
                    piece.coef_y.push_back(c(1, k));
                    piece.coef_s.push_back(c1(0, k));
                }
                trajSegment.trajs.push_back(piece);
                accum_t += piece_dt;
            }

            if (!trajSegment.trajs.empty())
            {
                trajmsg.traj_container.push_back(trajSegment);
                trajmsg.reverse.push_back(singual <= 0);
            }
            else
            {
                break;
            }

            if (accum_t >= publish_duration - 1e-6)
            {
                break;
            }
        }

        if (trajmsg.traj_container.empty())
        {
            RCLCPP_WARN(node_->get_logger(),
                        "No safe trajectory pieces available to publish (partial=%d, duration=%.2f)",
                        traj_publish_partial_active_ ? 1 : 0, publish_duration);
            traj_publish_partial_active_ = false;
            traj_publish_partial_duration_ = -1.0;
            return;
        }
        
        published_traj_duration_ = std::max(0.0, accum_t);
        trajCmdPub->publish(trajmsg);
        if (traj_publish_partial_active_)
        {
            RCLCPP_WARN(node_->get_logger(),
                        "Published PARTIAL trajectory with %d segments (%.2fs / %.2fs)",
                        static_cast<int>(trajmsg.traj_container.size()), publish_duration, total_duration);
        }
        else
        {
            RCLCPP_DEBUG(node_->get_logger(), "Published trajectory with %d segments", (int)trajmsg.traj_container.size());
        }
        traj_publish_partial_active_ = false;
        traj_publish_partial_duration_ = -1.0;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Exception in publishSimTraj2Controller: %s", e.what());
        traj_publish_partial_active_ = false;
        traj_publish_partial_duration_ = -1.0;
    }
}
double PlanManager::getDuration(){
    std::lock_guard<std::mutex> lock(traj_mutex_);
    const double total_duration = optTraj.getTotalDuration();
    if (published_traj_duration_ > 0.0)
        return std::min(total_duration, published_traj_duration_);
    return total_duration;
}

void PlanManager::setInitStateAndInput(const Eigen::Vector4d& state, const double& start_time){
    start_state_ = state;
    start_time_ = start_time;
    hasOdom = true;  // 标记接收到里程计
    RCLCPP_DEBUG(node_->get_logger(), "Init state set: [%.2f, %.2f, %.2f, %.2f]", state[0], state[1], state[2], state[3]);
}
double PlanManager::getTrajStartTime(){
    std::lock_guard<std::mutex> lock(traj_mutex_);
    return optTraj.start_time;
}

void PlanManager::setTrajStartTime(const double& traj_start_time){
    start_time_ = traj_start_time;
    std::lock_guard<std::mutex> lock(traj_mutex_);
    optTraj.start_time = traj_start_time;
}

void PlanManager::setParkingEnd(Eigen::Vector4d end_pt){
    end_state_ = end_pt;
    hasTarget = true;
    RCLCPP_INFO(node_->get_logger(), "Target set: [%.2f, %.2f, %.2f, %.2f]", end_pt[0], end_pt[1], end_pt[2], end_pt[3]);
}

bool PlanManager::getPredictState(const double pre_t, Eigen::Vector4d& state){
    std::lock_guard<std::mutex> lock(traj_mutex_);
    if(optTraj.Traj_container.empty()){
        RCLCPP_DEBUG(node_->get_logger(), "No trajectory available");
        return false;
    }
    
    double t = pre_t - optTraj.start_time;
    double total_duration = optTraj.getTotalDuration();
    if (published_traj_duration_ > 0.0)
        total_duration = std::min(total_duration, published_traj_duration_);
    
    if(t < 0.0) t = 0.0;
    if(t > total_duration) t = total_duration; // Clamp instead of failing to allow FSM to compute end-of-path tracking error
    
    try {
        state.head(2) = optTraj.getPos(t);
        state(2) = optTraj.getYaw(t);
        state(3) = optTraj.getVelItem(t);
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Error getting predict state: %s", e.what());
        return false;
    }
}

void PlanManager::clearVisualizationPath(){
    std::lock_guard<std::mutex> viz_lock(viz_mutex_);
    hasVisPath = false;
    lastVisPath.poses.clear();
    lastVisPath.header.frame_id = "map";
    lastVisPath.header.stamp = node_->get_clock()->now();
    if(publish_optimized_path_visualization_ && visPathPub != nullptr){
        visPathPub->publish(lastVisPath);
    }
}

void PlanManager::republishVisualizationPath(){
    std::lock_guard<std::mutex> viz_lock(viz_mutex_);
    if(!publish_optimized_path_visualization_ || !hasVisPath || visPathPub == nullptr){
        return;
    }
    lastVisPath.header.stamp = node_->get_clock()->now();
    visPathPub->publish(lastVisPath);
}

void PlanManager::republishTrajectory()
{
    // DO NOT lock traj_mutex_ here! publishSimTraj2Controller() already locks it internally.
    // Nested locking of std::mutex causes a complete thread deadlock!
    publishSimTraj2Controller();

    // Also keep visualization alive
    std::lock_guard<std::mutex> viz_lock(viz_mutex_);
    if (publish_optimized_path_visualization_ && hasVisPath && visPathPub != nullptr)
    {
        lastVisPath.header.stamp = node_->get_clock()->now();
        visPathPub->publish(lastVisPath);
    }
}
