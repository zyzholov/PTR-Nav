#include "mpc/mpc.h"
#include <sstream>
 
using namespace std;

namespace {

double wrapToPiMpc(double ang)
{
    while (ang > M_PI) ang -= 2.0 * M_PI;
    while (ang < -M_PI) ang += 2.0 * M_PI;
    return ang;
}

double cross2dMpc(const Eigen::Vector2d &a, const Eigen::Vector2d &b)
{
    return a.x() * b.y() - a.y() * b.x();
}

bool onSegment2dMpc(const Eigen::Vector2d &a,
                    const Eigen::Vector2d &b,
                    const Eigen::Vector2d &p)
{
    constexpr double eps = 1.0e-6;
    return std::min(a.x(), b.x()) - eps <= p.x() && p.x() <= std::max(a.x(), b.x()) + eps &&
           std::min(a.y(), b.y()) - eps <= p.y() && p.y() <= std::max(a.y(), b.y()) + eps &&
           std::fabs(cross2dMpc(b - a, p - a)) <= eps;
}

bool segmentsIntersect2dMpc(const Eigen::Vector2d &a,
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

    const double o1 = cross2dMpc(b - a, c - a);
    const double o2 = cross2dMpc(b - a, d - a);
    const double o3 = cross2dMpc(d - c, a - c);
    const double o4 = cross2dMpc(d - c, b - c);

    if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
        ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps)))
    {
        return true;
    }

    return (std::fabs(o1) <= eps && onSegment2dMpc(a, b, c)) ||
           (std::fabs(o2) <= eps && onSegment2dMpc(a, b, d)) ||
           (std::fabs(o3) <= eps && onSegment2dMpc(c, d, a)) ||
           (std::fabs(o4) <= eps && onSegment2dMpc(c, d, b));
}

bool validateTrajPointLocalGeometry(const std::vector<TrajPoint> &raw_wps,
                                    std::string &reason)
{
    reason.clear();

    std::vector<Eigen::Vector2d> pts;
    pts.reserve(raw_wps.size());
    for (const auto &wp : raw_wps)
    {
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y))
        {
            reason = "non-finite point";
            return false;
        }
        Eigen::Vector2d p(wp.x, wp.y);
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
            const double turn = std::fabs(wrapToPiMpc(seg_yaw[i] - seg_yaw[prev_valid_seg]));
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
            if (segmentsIntersect2dMpc(pts[i], pts[i + 1], pts[j], pts[j + 1]))
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
 
void MPC::init(std::shared_ptr<rclcpp::Node> node)
{
    node_shared_ = node;
    node_ = node.get();  // Save non-owning node pointer for later use
    vector<double> Q_ = {10, 10, 0.5};
    vector<double> R_ = {10, 10, 0.5};
    vector<double> Rd_ = {10, 10, 0.5};
    node->declare_parameter("mpc/dt", -1.0);
    node->declare_parameter("mpc/predict_steps", -1);
    node->declare_parameter("mpc/max_steer", -1.0);
    node->declare_parameter("mpc/max_dsteer", -1.0);
    node->declare_parameter("mpc/max_speed", -1.0);
    node->declare_parameter("mpc/min_speed", -1.0);
    node->declare_parameter("mpc/reverse_speed_cap", 0.6);
    node->declare_parameter("mpc/max_accel", -1.0);
    node->declare_parameter("mpc/wheel_base", -1.0);
    node->declare_parameter("mpc/delay_num", -1);
    node->declare_parameter("mpc/test_mpc", false);
    node->declare_parameter("mpc/fallback_block_after_planner_sec", 2.0);
    node->declare_parameter("mpc/enable_fallback_goal", false);
    node->declare_parameter("mpc/cmd_is_yaw_rate", false);
    node->declare_parameter("mpc/use_nmpc_solver", true);
    node->declare_parameter("mpc/enable_aggressive_recovery", false);
    node->declare_parameter("mpc/force_reverse_follow", true);
    node->declare_parameter("mpc/reverse_follow_speed_floor", 0.12);
    node->declare_parameter("mpc/reverse_hint_hold_sec", 3.0);
    node->declare_parameter("mpc/planner_update_min_interval_sec", 1.2);
    node->declare_parameter("mpc/planner_update_max_jump_m", 2.0);
    node->declare_parameter("mpc/replan_cautious_sec", 1.0);
    node->declare_parameter("mpc/replan_cautious_speed_cap", 0.14);
    node->declare_parameter("mpc/replan_cautious_dist_thresh", 0.50);
    node->declare_parameter("mpc/replan_cautious_yaw_thresh", 0.55);
    node->declare_parameter("mpc/replan_cautious_steer_thresh", 0.24);
    node->declare_parameter("mpc/continuous_replan_mode", false);
    node->declare_parameter("mpc/cmd_speed_scale", 1.0);
    node->declare_parameter("mpc/turn_speed_steer_gain", 0.55);
    node->declare_parameter("mpc/turn_speed_min_cap", 0.12);
    node->declare_parameter("mpc/ref_projection_period_sec", 0.08);
    node->declare_parameter("mpc/control_visualization_period_sec", 0.50);
    node->declare_parameter("mpc/matrix_q", std::vector<double>());
    node->declare_parameter("mpc/matrix_r", std::vector<double>());
    node->declare_parameter("mpc/matrix_rd", std::vector<double>());
    
    dt = node->get_parameter("mpc/dt").as_double();
    Npre = node->get_parameter("mpc/predict_steps").as_int();
    max_steer = node->get_parameter("mpc/max_steer").as_double();
    max_dsteer = node->get_parameter("mpc/max_dsteer").as_double();
    max_speed = node->get_parameter("mpc/max_speed").as_double();
    min_speed = node->get_parameter("mpc/min_speed").as_double();
    reverse_speed_cap = node->get_parameter("mpc/reverse_speed_cap").as_double();
    max_accel = node->get_parameter("mpc/max_accel").as_double();
    wheel_base = node->get_parameter("mpc/wheel_base").as_double();
    delay_num = node->get_parameter("mpc/delay_num").as_int();
    test_mpc = node->get_parameter("mpc/test_mpc").as_bool();
    fallback_block_after_planner_sec_ = node->get_parameter("mpc/fallback_block_after_planner_sec").as_double();
    enable_fallback_goal_ = node->get_parameter("mpc/enable_fallback_goal").as_bool();
    cmd_is_yaw_rate_ = node->get_parameter("mpc/cmd_is_yaw_rate").as_bool();
    use_nmpc_solver_ = node->get_parameter("mpc/use_nmpc_solver").as_bool();
    enable_aggressive_recovery_ = node->get_parameter("mpc/enable_aggressive_recovery").as_bool();
    force_reverse_follow_ = node->get_parameter("mpc/force_reverse_follow").as_bool();
    reverse_follow_speed_floor_ = node->get_parameter("mpc/reverse_follow_speed_floor").as_double();
    reverse_hint_hold_sec_ = node->get_parameter("mpc/reverse_hint_hold_sec").as_double();
    planner_update_min_interval_sec_ = node->get_parameter("mpc/planner_update_min_interval_sec").as_double();
    planner_update_max_jump_m_ = node->get_parameter("mpc/planner_update_max_jump_m").as_double();
    replan_cautious_sec_ = node->get_parameter("mpc/replan_cautious_sec").as_double();
    replan_cautious_speed_cap_ = node->get_parameter("mpc/replan_cautious_speed_cap").as_double();
    replan_cautious_dist_thresh_ = node->get_parameter("mpc/replan_cautious_dist_thresh").as_double();
    replan_cautious_yaw_thresh_ = node->get_parameter("mpc/replan_cautious_yaw_thresh").as_double();
    replan_cautious_steer_thresh_ = node->get_parameter("mpc/replan_cautious_steer_thresh").as_double();
    cmd_speed_scale_ = node->get_parameter("mpc/cmd_speed_scale").as_double();
    turn_speed_steer_gain_ = node->get_parameter("mpc/turn_speed_steer_gain").as_double();
    turn_speed_min_cap_ = node->get_parameter("mpc/turn_speed_min_cap").as_double();
    ref_projection_period_sec_ = node->get_parameter("mpc/ref_projection_period_sec").as_double();
    control_visualization_period_sec_ = node->get_parameter("mpc/control_visualization_period_sec").as_double();
    continuous_replan_mode_ = node->get_parameter("mpc/continuous_replan_mode").as_bool();
    if (continuous_replan_mode_)
    {
        RCLCPP_INFO(node_->get_logger(), "MPC continuous replan mode enabled — relaxed validation, trajectory blending active");
    }
    Q_ = node->get_parameter("mpc/matrix_q").as_double_array();
    R_ = node->get_parameter("mpc/matrix_r").as_double_array();
    Rd_ = node->get_parameter("mpc/matrix_rd").as_double_array();

    if (dt <= 0.0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/dt invalid (%.3f), using default 0.1", dt);
        dt = 0.1;
    }
    if (Npre < 2) {
        RCLCPP_WARN(node_->get_logger(), "mpc/predict_steps invalid (%d), using default 10", Npre);
        Npre = 10;
    }
    if (delay_num < 0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/delay_num invalid (%d), using 0", delay_num);
        delay_num = 0;
    }
    if (wheel_base <= 1e-3) {
        RCLCPP_WARN(node_->get_logger(), "mpc/wheel_base invalid (%.3f), using 0.6", wheel_base);
        wheel_base = 0.6;
    }

    if (max_steer <= 0.0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/max_steer invalid (%.3f), using default 0.7 rad", max_steer);
        max_steer = 0.7;
    }
    if (max_steer > 1.2) {
        RCLCPP_WARN(node_->get_logger(), "mpc/max_steer too large (%.3f), clamped to 1.2 rad for stability", max_steer);
        max_steer = 1.2;
    }
    if (max_speed <= min_speed) {
        RCLCPP_WARN(node_->get_logger(), "Invalid speed bounds (min=%.3f max=%.3f), reset to [-1.0, 1.5]", min_speed, max_speed);
        min_speed = -1.0;
        max_speed = 1.5;
    }
    // Hard safety caps to stay consistent even when external parameter overlays differ.
    max_speed = std::min(max_speed, 2.5);
    min_speed = std::max(min_speed, -0.6);
    if (max_dsteer <= 0.0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/max_dsteer invalid (%.3f), using default 0.6", max_dsteer);
        max_dsteer = 0.6;
    }
    if (max_accel <= 0.0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/max_accel invalid (%.3f), using default 1.0", max_accel);
        max_accel = 1.0;
    }
    if (reverse_speed_cap < 0.0) {
        RCLCPP_WARN(node_->get_logger(), "mpc/reverse_speed_cap invalid (%.3f), using 0.6", reverse_speed_cap);
        reverse_speed_cap = 0.6;
    }
    reverse_follow_speed_floor_ = std::clamp(reverse_follow_speed_floor_, 0.05, 0.30);
    reverse_hint_hold_sec_ = std::clamp(reverse_hint_hold_sec_, 0.5, 8.0);
    replan_cautious_sec_ = std::clamp(replan_cautious_sec_, 0.0, 3.0);
    replan_cautious_speed_cap_ = std::clamp(replan_cautious_speed_cap_, 0.08, 0.45);
    replan_cautious_dist_thresh_ = std::clamp(replan_cautious_dist_thresh_, 0.20, 1.20);
    replan_cautious_yaw_thresh_ = std::clamp(replan_cautious_yaw_thresh_, 0.20, 1.20);
    replan_cautious_steer_thresh_ = std::clamp(replan_cautious_steer_thresh_, 0.05, max_steer);
    cmd_speed_scale_ = std::clamp(cmd_speed_scale_, 0.5, 3.0);
    turn_speed_steer_gain_ = std::clamp(turn_speed_steer_gain_, 0.20, 0.90);
    turn_speed_min_cap_ = std::clamp(turn_speed_min_cap_, 0.06, 1.00);
    ref_projection_period_sec_ = std::clamp(ref_projection_period_sec_, 0.04, 0.25);
    control_visualization_period_sec_ = std::clamp(control_visualization_period_sec_, 0.0, 2.0);
    if (turn_speed_min_cap_ > max_speed) {
        RCLCPP_WARN(node_->get_logger(),
                    "mpc/turn_speed_min_cap (%.3f) exceeds mpc/max_speed (%.3f), clamped to max_speed",
                    turn_speed_min_cap_, max_speed);
        turn_speed_min_cap_ = max_speed;
    }

    // Defensive checks: ensure parameter vectors have expected sizes
    if (Q_.size() < 3) {
        RCLCPP_WARN(node_->get_logger(), "mpc/matrix_q has size %zu, expected 3. Using default.", Q_.size());
        Q_ = {100.0, 100.0, 10.0};
    }
    if (R_.size() < 2) {
        RCLCPP_WARN(node_->get_logger(), "mpc/matrix_r has size %zu, expected 2. Using default.", R_.size());
        R_ = {0.1, 0.01};
    }
    if (Rd_.size() < 2) {
        RCLCPP_WARN(node_->get_logger(), "mpc/matrix_rd has size %zu, expected 2. Using default.", Rd_.size());
        Rd_ = {5.0, 10.0};
    }

    q_diag_ = {Q_[0], Q_[1], Q_[2]};
    r_diag_ = {R_[0], R_[1]};
    rd_diag_ = {Rd_[0], Rd_[1]};
    solver_buf_inited_ = false;
    casadi_param_inited_ = false;
    RCLCPP_INFO(node_->get_logger(), "MPC init done: dt=%.2f, Npre=%d, v=[%.2f, %.2f]", dt, Npre, min_speed, max_speed);

    RCLCPP_INFO(node_->get_logger(), "MPC init stage: declare bk params");
    node->declare_parameter("mpc/bk_mode", false);
    node->declare_parameter("mpc/traj_file", std::string("xxx"));
    bk_mode = node->get_parameter("mpc/bk_mode").as_bool();
    traj_file = node->get_parameter("mpc/traj_file").as_string();

    has_odom = false;
    receive_traj = false;
    planner_traj_valid_ = false;
    planner_traj_stamp_ = node_->get_clock()->now();
    planner_traj_last_accept_stamp_ = node_->get_clock()->now();
    reverse_allowed_ = false;
    reverse_hold_until_sec_ = node_->get_clock()->now().seconds();
    traj_prefers_reverse_ = false;
    traj_reverse_ratio_ = 0.0;
    traj_reverse_hint_until_sec_ = reverse_hold_until_sec_;
    ref_tracking_valid_ = false;
    ref_tracking_time_ = 0.0;
    last_cmd_callback_time_sec_ = 0.0;
    last_cmd_publish_wall_sec_ = 0.0;
    cmd_diag_window_start_wall_sec_ = 0.0;
    cmd_diag_max_gap_sec_ = 0.0;
    cmd_diag_max_dv_ = 0.0;
    cmd_diag_max_ds_ = 0.0;
    cmd_diag_publish_count_ = 0;
    last_ref_projection_sec_ = 0.0;
    last_follow_vis_pub_sec_ = 0.0;
    last_control_vis_pub_sec_ = 0.0;
    output_buff.assign(static_cast<size_t>(delay_num), Eigen::Vector2d::Zero());
    cmd.linear.x = 0.0;
    cmd.angular.z = 0.0;

    // Initialize trajectory analyzers with node's clock
    RCLCPP_INFO(node_->get_logger(), "MPC init stage: init traj analyzers");
    traj_analyzer.setNode(node);
    new_traj_analyzer.setNode(node);

    RCLCPP_INFO(node_->get_logger(), "MPC init stage: create publishers");
    auto cmd_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    auto vis_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    pos_cmd_pub_ = node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", cmd_qos);
    vis_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/following_path", vis_qos);
    predict_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/predict_path", vis_qos);
    ref_pub = node->create_publisher<visualization_msgs::msg::MarkerArray>("/reference_path", vis_qos);
    history_pub = node->create_publisher<visualization_msgs::msg::Marker>("/vehicle_history", vis_qos);
    desiredS_pub_ = node->create_publisher<mpc_controller::msg::DesiredState>("/desired_state",10);
    clearVisualization();
    
    RCLCPP_INFO(node_->get_logger(), "MPC init stage: create timer/subscribers");
    timer_cb_group_ = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    odom_cb_group_ = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    traj_cb_group_ = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions odom_opts;
    odom_opts.callback_group = odom_cb_group_;
    rclcpp::SubscriptionOptions traj_opts;
    traj_opts.callback_group = traj_cb_group_;

    cmd_timer_ = node->create_wall_timer(
        std::chrono::milliseconds(20),
        std::bind(&MPC::cmdCallback, this),
        timer_cb_group_);
    odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
        "odom", rclcpp::QoS(1), std::bind(&MPC::rcvOdomCallBack, this, std::placeholders::_1), odom_opts);
    traj_sub_1 = node->create_subscription<mpc_controller::msg::SE2Traj>(
        "se2traj", rclcpp::QoS(1), std::bind(&MPC::rcvTrajCallBack, this, std::placeholders::_1), traj_opts);
    traj_sub_2 = node->create_subscription<mpc_controller::msg::PolyTraj>(
        "traj", rclcpp::QoS(1), std::bind(&MPC::rcvFlatTrajCallBack, this, std::placeholders::_1), traj_opts);
    traj_sub_3 = node->create_subscription<mpc_controller::msg::DPtrajContainer>(
        "dptraj", rclcpp::QoS(1), std::bind(&MPC::rcvDPTrajCallBack, this, std::placeholders::_1), traj_opts);
    err_pub = node->create_publisher<std_msgs::msg::Float64>("/track_err", 10);
    trigger_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", rclcpp::QoS(1), std::bind(&MPC::rcvTriggerCallBack, this, std::placeholders::_1), traj_opts);
    trigger_sub_legacy_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", rclcpp::QoS(1), std::bind(&MPC::rcvTriggerCallBack, this, std::placeholders::_1), traj_opts);
    // Initialize CasADi matrices and pre-build NLP before the control loop.
    // Codegen+JIT takes 100-200ms — doing it now prevents control gaps later.
    if (!casadi_param_inited_)
    {
        Q = DM::zeros(3, 3);
        Q(0, 0) = q_diag_[0]; Q(1, 1) = q_diag_[1]; Q(2, 2) = q_diag_[2];
        R = DM::zeros(2, 2);
        R(0, 0) = r_diag_[0]; R(1, 1) = r_diag_[1];
        Rd = DM::zeros(2, 2);
        Rd(0, 0) = rd_diag_[0]; Rd(1, 1) = rd_diag_[1];
        U_min = DM::zeros(2, 1); U_max = DM::zeros(2, 1);
        U_min(0, 0) = std::max(min_speed, -reverse_speed_cap); U_min(1, 0) = -max_steer;
        U_max(0, 0) = max_speed; U_max(1, 0) = max_steer;
        casadi_param_inited_ = true;
    }
    if (use_nmpc_solver_ && !nlp_built_)
        buildNLP();
    if (!use_nmpc_solver_)
    {
        X_sol = DM::zeros(3, Npre);
        U_sol = DM::zeros(2, Npre);
        u_0 = DM::zeros(2, 1);
    }
    RCLCPP_INFO(node_->get_logger(), "MPC trigger subscribers active on /goal_pose and /move_base_simple/goal");
    RCLCPP_INFO(node_->get_logger(), "MPC fallback goal is %s", enable_fallback_goal_ ? "enabled" : "disabled");
    RCLCPP_INFO(node_->get_logger(), "MPC cmd angular mode: %s", cmd_is_yaw_rate_ ? "yaw_rate" : "steer_angle");
    RCLCPP_INFO(node_->get_logger(), "MPC CasADi SQP solver: %s", use_nmpc_solver_ ? "enabled" : "disabled");
    RCLCPP_INFO(node_->get_logger(), "MPC aggressive recovery: %s", enable_aggressive_recovery_ ? "enabled" : "disabled");
    RCLCPP_INFO(node_->get_logger(), "MPC reverse follow: %s (floor=%.2f hold=%.2fs)",
                force_reverse_follow_ ? "enabled" : "disabled",
                reverse_follow_speed_floor_, reverse_hint_hold_sec_);
    RCLCPP_INFO(node_->get_logger(), "MPC command speed scale: %.2f", cmd_speed_scale_);
    RCLCPP_INFO(node_->get_logger(), "MPC turn-speed cap: steer_gain=%.2f min_cap=%.2f",
                turn_speed_steer_gain_, turn_speed_min_cap_);
    RCLCPP_INFO(node_->get_logger(), "MPC ROS interfaces initialized");

    errs.clear();
    vels.clear();
    angle_vels.clear();
    jerks.clear();
    errs_yaw.clear();
    errs_lon.clear();
    errs_lat.clear();

    trajectory_length = 0.0;

    if (bk_mode)
    {
        string o = traj_file;
        outfile.open(o.insert(o.find("trajs"), "err_"), std::ofstream::out);
        outfile.clear();
    }
}

void MPC::rcvTriggerCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!enable_fallback_goal_)
    {
        RCLCPP_DEBUG(node_->get_logger(), "Ignore fallback trigger because mpc/enable_fallback_goal=false");
        return;
    }

    auto now_t = node_->get_clock()->now();
    bool block_fallback = false;
    double planner_age = 0.0;
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        planner_age = (now_t - planner_traj_stamp_).seconds();
        block_fallback = planner_traj_valid_ && !traj_analyzer.at_goal &&
                         planner_age < fallback_block_after_planner_sec_;
    }
    if (block_fallback)
    {
        RCLCPP_INFO(node_->get_logger(),
                    "Skip fallback goal (%.2f, %.2f): planner trajectory is fresh (%.2fs ago)",
                    msg->pose.position.x, msg->pose.position.y,
                    planner_age);
        return;
    }

    RCLCPP_INFO(node_->get_logger(), "Received goal_pose (%.2f, %.2f). Generating fallback trajectory since planner unavailable...", 
                msg->pose.position.x, msg->pose.position.y);
    
    MPCState odom_snapshot;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        if (!has_odom) {
            RCLCPP_WARN(node_->get_logger(), "  No odometry yet; cannot generate fallback");
            return;
        }
        odom_snapshot = now_state;
    }
    // Generate a simple 4-point SE2Traj from current pose to goal
    auto fallback = std::make_shared<mpc_controller::msg::SE2Traj>();
    fallback->start_time = node_->get_clock()->now();
    
    geometry_msgs::msg::Point p0, p1, p2, p3;
    p0.x = odom_snapshot.x;
    p0.y = odom_snapshot.y;
    p0.z = 0.0;
    
    p3 = msg->pose.position;
    p3.z = 0.0;
    
    p1.x = p0.x + (p3.x - p0.x) * 0.33;
    p1.y = p0.y + (p3.y - p0.y) * 0.33;
    p1.z = 0.0;
    
    p2.x = p0.x + (p3.x - p0.x) * 0.67;
    p2.y = p0.y + (p3.y - p0.y) * 0.67;
    p2.z = 0.0;
    
    fallback->pos_pts = {p0, p1, p2, p3};
    
    // Calculate total distance and plan time accordingly (at ~1 m/s nominal speed)
    double dx = p3.x - p0.x;
    double dy = p3.y - p0.y;
    double total_dist = sqrt(dx*dx + dy*dy);
    double total_time = std::max(3.0, total_dist * 1.5);  // At least 3 sec, or 1.5 sec per meter
    
    double target_yaw = atan2(p3.y - p0.y, p3.x - p0.x);
    geometry_msgs::msg::Point a0, a1, a2, a3;
    a0.x = odom_snapshot.theta;
    a1.x = odom_snapshot.theta + (target_yaw - odom_snapshot.theta) * 0.33;
    a2.x = odom_snapshot.theta + (target_yaw - odom_snapshot.theta) * 0.67;
    a3.x = target_yaw;
    
    fallback->angle_pts = {a0, a1, a2, a3};
    fallback->init_v.x = 0.0;
    fallback->init_v.y = 0.0;
    fallback->init_v.z = 0.0;
    fallback->init_a.x = 0.0;
    fallback->init_a.y = 0.0;
    fallback->init_a.z = 0.0;
    
    // Distribute time evenly across 4 segments
    double seg_time = total_time / 4.0;
    fallback->pos_t_pts = {seg_time, seg_time, seg_time, seg_time};
    fallback->angle_t_pts = {seg_time, seg_time, seg_time, seg_time};
    
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        new_traj_analyzer.setTraj(fallback);
        traj_analyzer = new_traj_analyzer;
        receive_traj = true;
        has_output = false;
        ref_tracking_valid_ = false;
        ref_tracking_time_ = 0.0;
        last_ref_projection_sec_ = 0.0;
    }
    
    RCLCPP_INFO(node_->get_logger(), "✓ Fallback trajectory ready: (%.2f, %.2f) -> (%.2f, %.2f), distance=%.2f, time=%.2f sec", 
                p0.x, p0.y, p3.x, p3.y, total_dist, total_time);
}

void MPC::rcvTrajCallBack(const mpc_controller::msg::SE2Traj::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(traj_mutex_);
    traj_analyzer.setTraj(msg);
    ref_tracking_valid_ = false;
    ref_tracking_time_ = 0.0;
    last_ref_projection_sec_ = 0.0;
    latched_ref_path.clear();
    auto wps = traj_analyzer.getTajWps(0.2);
    const double vis_ref_z = has_odom ? (now_state.z + 0.05) : 0.15;
    latched_ref_path.reserve(wps.size());
    for (const auto &p : wps)
    {
        geometry_msgs::msg::Point pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = vis_ref_z;
        latched_ref_path.push_back(pt);
    }
    if (!receive_traj || traj_analyzer.at_goal)
    {
        mission_history.clear();
    }
    receive_traj = true;
}

void MPC::rcvFlatTrajCallBack(const mpc_controller::msg::PolyTraj::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(traj_mutex_);
    traj_analyzer.setTraj(msg);
    ref_tracking_valid_ = false;
    ref_tracking_time_ = 0.0;
    last_ref_projection_sec_ = 0.0;
    latched_ref_path.clear();
    auto wps = traj_analyzer.getTajWps(0.2);
    const double vis_ref_z = has_odom ? (now_state.z + 0.05) : 0.15;
    latched_ref_path.reserve(wps.size());
    for (const auto &p : wps)
    {
        geometry_msgs::msg::Point pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = vis_ref_z;
        latched_ref_path.push_back(pt);
    }
    if (!receive_traj || traj_analyzer.at_goal)
    {
        mission_history.clear();
    }
    receive_traj = true;
    has_output = false;
}
void MPC::rcvDPTrajCallBack(const mpc_controller::msg::DPtrajContainer::SharedPtr msg){
    auto now_t = node_->get_clock()->now();
    if (msg->traj_container.empty())
    {
        RCLCPP_WARN(node_->get_logger(), "Ignore empty DPtrajContainer from planner");
        return;
    }

    TrajAnalyzer candidate_traj;
    candidate_traj.setNode(node_shared_);
    candidate_traj.setTraj(msg);
    RCLCPP_DEBUG(node_->get_logger(), "Received DPtrajContainer: %zu seg, dur=%.2fs ts=%u.%09u",
                 msg->traj_container.size(), candidate_traj.getDuration(),
                 msg->start_time.sec, msg->start_time.nanosec);

    // Sanity-check planner trajectory before accepting it as active reference.
    bool traj_valid = true;
    std::string invalid_reason;
    std::vector<TrajPoint> preview;
    try {
        preview = candidate_traj.getRefPoints(std::max(6, Npre), dt);
    } catch (const std::exception &e) {
        traj_valid = false;
        invalid_reason = std::string("getRefPoints exception: ") + e.what();
    }

    if (traj_valid && preview.empty()) {
        traj_valid = false;
        invalid_reason = "empty preview points";
    }

    if (traj_valid) {
        try {
            const double geom_dt = std::max(0.08, std::min(0.18, dt));
            auto geom_wps = candidate_traj.getTajWps(geom_dt);
            std::string geom_reason;
            if (!validateTrajPointLocalGeometry(geom_wps, geom_reason)) {
                traj_valid = false;
                invalid_reason = std::string("bad local trajectory geometry: ") + geom_reason;
            }
        } catch (const std::exception &e) {
            traj_valid = false;
            invalid_reason = std::string("geometry validation exception: ") + e.what();
        }
    }

    const double cur_limit = std::max(0.2, std::tan(std::max(0.05, max_steer)) / std::max(1e-3, wheel_base) * 2.0);
    double max_abs_cur = 0.0;
    if (traj_valid) {
        for (size_t i = 0; i < preview.size(); ++i) {
            const auto &p = preview[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.theta) ||
                !std::isfinite(p.v) || !std::isfinite(p.cur)) {
                traj_valid = false;
                invalid_reason = "non-finite preview state";
                break;
            }
            max_abs_cur = std::max(max_abs_cur, std::abs(p.cur));
        }
    }

    if (traj_valid) {
        const double hard_cur_factor = continuous_replan_mode_ ? 5.0 : 3.0;
        const double hard_cur_limit = cur_limit * hard_cur_factor;
        if (max_abs_cur > hard_cur_limit) {
            traj_valid = false;
            invalid_reason = "curvature far beyond trackable range";
        } else if (max_abs_cur > cur_limit) {
            RCLCPP_WARN(node_->get_logger(),
                        "Planner trajectory curvature is high (max_cur=%.3f, soft_limit=%.3f), keep it and rely on MPC speed/steer saturation",
                        max_abs_cur, cur_limit);
        }
    }

    MPCState odom_snapshot;
    bool have_odom_snapshot = false;
    {
        std::lock_guard<std::mutex> odom_lock(odom_mutex_);
        odom_snapshot = now_state;
        have_odom_snapshot = has_odom;
    }

    if (traj_valid && have_odom_snapshot) {
        const double jump_factor = continuous_replan_mode_ ? 1.35 : 1.0;
        const double jump_limit = std::max(0.55, planner_update_max_jump_m_ * jump_factor);
        const double jump_front = std::hypot(preview.front().x - odom_snapshot.x, preview.front().y - odom_snapshot.y);
        double jump_min = jump_front;
        for (const auto &p : preview) {
            const double d = std::hypot(p.x - odom_snapshot.x, p.y - odom_snapshot.y);
            jump_min = std::min(jump_min, d);
        }
        double nearest_yaw_gap = 0.0;
        try {
            const double nearest_t = candidate_traj.nearestTimeToPublic(odom_snapshot.x, odom_snapshot.y, 0.05);
            auto near_pts = candidate_traj.getRefPointsFromTime(1, dt, nearest_t, odom_snapshot.x, odom_snapshot.y);
            if (!near_pts.empty())
            {
                nearest_yaw_gap = near_pts.front().theta - odom_snapshot.theta;
                while (nearest_yaw_gap > M_PI) nearest_yaw_gap -= 2.0 * M_PI;
                while (nearest_yaw_gap < -M_PI) nearest_yaw_gap += 2.0 * M_PI;
            }
        } catch (...) {}

        if (jump_min > jump_limit) {
            traj_valid = false;
            invalid_reason = "reference preview jumps too far";
        } else if (jump_min < 0.55 && std::abs(nearest_yaw_gap) > 0.95) {
            traj_valid = false;
            invalid_reason = "reference nearest yaw mismatch";
        } else if (jump_front > jump_limit) {
            RCLCPP_WARN(node_->get_logger(),
                        "Planner trajectory front jump=%.2f exceeds limit=%.2f (min_jump=%.2f yaw_gap=%.2f), keep nearest-continuous trajectory",
                        jump_front, jump_limit, jump_min, nearest_yaw_gap);
        }
    }

    if (!traj_valid) {
        RCLCPP_WARN(node_->get_logger(), "Reject planner trajectory: %s (fallback trajectory kept)", invalid_reason.c_str());
        return;
    }

    // Extract reverse-duration ratio from planner DP trajectory so controller can
    // intentionally follow reverse-dominant maneuvers instead of stalling at zero speed.
    double total_seg_time = 0.0;
    double reverse_seg_time = 0.0;
    for (size_t si = 0; si < msg->traj_container.size(); ++si)
    {
        double seg_dt = 0.0;
        for (const auto &piece : msg->traj_container[si].trajs)
        {
            seg_dt += std::max(0.0, static_cast<double>(piece.dt));
        }
        if (seg_dt <= 1e-3)
        {
            continue;
        }
        total_seg_time += seg_dt;
        const bool seg_reverse = (si < msg->reverse.size()) ? msg->reverse[si] : false;
        if (seg_reverse)
        {
            reverse_seg_time += seg_dt;
        }
    }
    const double new_reverse_ratio = (total_seg_time > 1e-3) ? (reverse_seg_time / total_seg_time) : 0.0;
    bool new_prefers_reverse = (reverse_seg_time > 0.6) && (new_reverse_ratio > 0.45);
    double new_reverse_hint_until_sec = now_t.seconds() + (new_prefers_reverse ? reverse_hint_hold_sec_ : 0.6);
    if (new_prefers_reverse)
    {
        RCLCPP_WARN(node_->get_logger(),
                    "Planner trajectory is reverse-dominant (reverse_ratio=%.2f, reverse_time=%.2fs), enable reverse-follow hint",
                    new_reverse_ratio, reverse_seg_time);
    }

    // If ANY segment is explicitly marked reverse, immediately allow reverse
    // following so the MPC does not stall when the planner commands backward motion.
    {
        bool has_reverse_seg = false;
        for (size_t si = 0; si < msg->reverse.size(); ++si)
        {
            if (msg->reverse[si])
            {
                has_reverse_seg = true;
                break;
            }
        }
        // In continuous mode, don't force reverse for individual segments.
        // Only the reverse-dominant check above (traj_reverse_ratio_ > 0.45)
        // should enable reverse.  Individual reverse segments are often
        // artifacts of path sampling near gear-switch points.
        if (has_reverse_seg && !continuous_replan_mode_)
            new_prefers_reverse = true;
    }

    std::vector<geometry_msgs::msg::Point> new_latched_ref_path;
    auto wps = candidate_traj.getTajWps(0.2);
    const double vis_ref_z = have_odom_snapshot ? (odom_snapshot.z + 0.05) : 0.15;
    new_latched_ref_path.reserve(wps.size());
    for (const auto &p : wps)
    {
        geometry_msgs::msg::Point pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = vis_ref_z;
        new_latched_ref_path.push_back(pt);
    }

    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        new_traj_analyzer = candidate_traj;
        latched_ref_path = std::move(new_latched_ref_path);
        if (!receive_traj || traj_analyzer.at_goal)
        {
            mission_history.clear();
        }
        traj_analyzer = new_traj_analyzer;
        receive_traj = true;
        planner_traj_valid_ = true;
        planner_traj_stamp_ = now_t;
        planner_traj_last_accept_stamp_ = now_t;
        traj_reverse_ratio_ = new_reverse_ratio;
        traj_prefers_reverse_ = new_prefers_reverse;
        traj_reverse_hint_until_sec_ = new_prefers_reverse
            ? std::max(new_reverse_hint_until_sec, now_t.seconds() + reverse_hint_hold_sec_ + 2.0)
            : new_reverse_hint_until_sec;
        if (new_prefers_reverse && !continuous_replan_mode_)
        {
            reverse_allowed_ = true;
            reverse_hold_until_sec_ = std::max(reverse_hold_until_sec_,
                                               now_t.seconds() + reverse_hint_hold_sec_ + 2.0);
        }
        has_output = false;

        if (have_odom_snapshot)
        {
            try {
                const double nearest_t = traj_analyzer.nearestTimeToPublic(odom_snapshot.x, odom_snapshot.y, 0.05);
                if (std::isfinite(nearest_t) && nearest_t >= 0.0)
                {
                    ref_tracking_time_ = nearest_t;
                    ref_tracking_valid_ = true;
                    last_ref_projection_sec_ = now_t.seconds();
                }
                else
                {
                    ref_tracking_valid_ = false;
                    ref_tracking_time_ = 0.0;
                    last_ref_projection_sec_ = 0.0;
                }
            } catch (...) {
                ref_tracking_valid_ = false;
                ref_tracking_time_ = 0.0;
                last_ref_projection_sec_ = 0.0;
            }
        }
        else
        {
            ref_tracking_valid_ = false;
            ref_tracking_time_ = 0.0;
            last_ref_projection_sec_ = 0.0;
        }
    }
}
void MPC::rcvOdomCallBack(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    MPCState odom_state;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        has_odom = true;
        trajectory_length += Eigen::Vector2d(now_state.x-msg->pose.pose.position.x, now_state.y-msg->pose.pose.position.y).norm();
        now_state.x = msg->pose.pose.position.x;
        now_state.y = msg->pose.pose.position.y;
        now_state.z = msg->pose.pose.position.z;
    {
        // Robust yaw extraction from full quaternion (works with non-zero roll/pitch on slopes).
        const double qw = msg->pose.pose.orientation.w;
        const double qx = msg->pose.pose.orientation.x;
        const double qy = msg->pose.pose.orientation.y;
        const double qz = msg->pose.pose.orientation.z;
        const double siny_cosp = 2.0 * (qw * qz + qx * qy);
        const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        now_state.theta = std::atan2(siny_cosp, cosy_cosp);
    }
        odom_state = now_state;
    }
    
    // Record vehicle position in history
    geometry_msgs::msg::Point pt;
    pt.x = odom_state.x;
    pt.y = odom_state.y;
    pt.z = odom_state.z + 0.08;
    vehicle_history.push_back(pt);

    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        if (receive_traj)
        {
            mission_history.push_back(pt);
            if (mission_history.size() > mission_history_max)
            {
                mission_history.erase(mission_history.begin());
            }
        }
    }
    
    // Keep history buffer manageable
    if (vehicle_history.size() > history_max) {
        vehicle_history.erase(vehicle_history.begin());
    }
}
void MPC::cmdCallback()
{
    if (cmd_running_.exchange(true))
        return;
    struct Guard { std::atomic_bool *p; ~Guard() { p->store(false); } } guard{&cmd_running_};

    MPCState state_snapshot;
    bool has_odom_snapshot = false;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        state_snapshot = now_state;
        has_odom_snapshot = has_odom;
    }

    if (!has_odom_snapshot)
    {
        return;
    }

    static int ref_gap_cycles = 0;
    auto reset_zero_command_memory = [&]() {
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.0;
        last_cmd_vel_ = 0.0;
        last_cmd_steer_ = 0.0;
        last_cmd_callback_time_sec_ = 0.0;
        if (!output_buff.empty())
            std::fill(output_buff.begin(), output_buff.end(), Eigen::Vector2d::Zero());
        u_0 = DM::zeros(2, 1);
    };
    auto release_active_traj = [&]() {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        receive_traj = false;
        planner_traj_valid_ = false;
        ref_tracking_valid_ = false;
        ref_tracking_time_ = 0.0;
        last_ref_projection_sec_ = 0.0;
    };
    auto publish_zero_command = [&]() {
        geometry_msgs::msg::Twist zero_cmd;
        pos_cmd_pub_->publish(zero_cmd);
        last_cmd_vel_ = 0.0;
        last_cmd_steer_ = 0.0;
    };

    bool receive_traj_snapshot = false;
    {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        receive_traj_snapshot = receive_traj;
    }

    if (!receive_traj_snapshot)
    {
        // Proactively publish zero command when trajectory is unavailable,
        // so downstream simulator/vehicle does not keep stale non-zero cmd.
        ref_gap_cycles = 0;
        reset_zero_command_memory();
        publish_zero_command();
        return;
    }

    auto now = node_->get_clock()->now();
    const double now_sec = now.seconds();

    double pred_x = state_snapshot.x;
    double pred_y = state_snapshot.y;
    double pred_yaw = state_snapshot.theta;
    auto wrap_pi = [](double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    };
    if (delay_num > 0 && !output_buff.empty() && wheel_base > 1e-3)
    {
        for (const auto &u : output_buff)
        {
            const double v = u.x();
            const double steer = std::clamp(u.y(), -max_steer, max_steer);
            pred_x += dt * v * std::cos(pred_yaw);
            pred_y += dt * v * std::sin(pred_yaw);
            pred_yaw += dt * v * std::tan(steer) / wheel_base;
            pred_yaw = wrap_pi(pred_yaw);
        }
    }
    bool traj_at_goal_snapshot = false;
    TrajPoint final_pt_snapshot;
    bool has_final_pt_snapshot = false;
    double traj_duration_snapshot = 0.0;
    double ref_tracking_time_snapshot = 0.0;

    // Get reference trajectory points.  DP trajectories come from the online
    // planner and their nominal time can run ahead of the physical vehicle when
    // the controller lags or the solver saturates.  In that case, anchor the
    // reference window to vehicle progress along the curve instead of letting it
    // jump to the terminal point and stall.
    try {
        std::lock_guard<std::mutex> lock(traj_mutex_);
        if (traj_analyzer.usesDpTrajectory())
        {
            const double duration = traj_analyzer.getDuration();
            traj_duration_snapshot = duration;
            const double sample_dt = std::max(0.05, std::min(0.15, dt));
            const bool projection_due =
                !ref_tracking_valid_ ||
                last_ref_projection_sec_ <= 0.0 ||
                now_sec < last_ref_projection_sec_ ||
                (now_sec - last_ref_projection_sec_) >= ref_projection_period_sec_;

            if (projection_due)
            {
                double nearest_t = 0.0;
                if (ref_tracking_valid_)
                {
                    const double elapsed = std::clamp(now_sec - last_ref_projection_sec_, 0.0, 0.35);
                    const double back_window = 0.18;
                    const double forward_window = std::max(0.55, elapsed + 0.45);
                    nearest_t = traj_analyzer.getNearestTimeInRange(
                        pred_x, pred_y, sample_dt,
                        ref_tracking_time_ - back_window,
                        ref_tracking_time_ + forward_window);

                    const double max_back_step = 0.05;
                    const double max_forward_step = std::max(0.10, elapsed + 0.12);
                    nearest_t = std::clamp(nearest_t,
                                           ref_tracking_time_ - max_back_step,
                                           ref_tracking_time_ + max_forward_step);
                }
                else
                {
                    nearest_t = traj_analyzer.getNearestTime(pred_x, pred_y, sample_dt);
                }

                ref_tracking_time_ = std::clamp(nearest_t, 0.0, std::max(0.0, duration));
                ref_tracking_valid_ = true;
                last_ref_projection_sec_ = now_sec;
            }
            double ref_window_start = ref_tracking_time_;
            if (last_ref_projection_sec_ > 0.0 && now_sec >= last_ref_projection_sec_)
            {
                const double elapsed_since_projection =
                    std::clamp(now_sec - last_ref_projection_sec_, 0.0,
                               std::max(0.02, ref_projection_period_sec_));
                ref_window_start = std::clamp(ref_tracking_time_ + elapsed_since_projection,
                                              0.0, std::max(0.0, duration));
            }
            ref_tracking_time_snapshot = ref_window_start;

            xref = traj_analyzer.getRefPointsFromTimeSpatial(Npre, 0.36, ref_window_start, pred_x, pred_y);
        }
        else
        {
            traj_duration_snapshot = traj_analyzer.getDuration();
            xref = traj_analyzer.getRefPoints(Npre, dt);
        }
        smooth_yaw(xref, state_snapshot.theta);
        traj_at_goal_snapshot = traj_analyzer.at_goal;
        if (traj_duration_snapshot > 1e-3)
        {
            final_pt_snapshot = traj_analyzer.getFinalPoint();
            has_final_pt_snapshot = true;
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(), "Error getting reference points: %s", e.what());
        return;
    }
    
    // --- 终点吸引 (Terminal Latching) ---
    // 如果前端规划器在终点前几十厘米提前截断导致 xref 为空，
    // 我们不直接停车，而是将物理终点作为静止参考点填入，把车辆平滑"吸"过去。
    if (xref.empty() && traj_at_goal_snapshot && has_final_pt_snapshot) {
        TrajPoint final_pt = final_pt_snapshot;
        final_pt.v = 0.0;
        final_pt.cur = 0.0;
        for (int i = 0; i < Npre; i++) {
            xref.push_back(final_pt);
        }
        smooth_yaw(xref, state_snapshot.theta);
    }

    if (xref.empty())
    {
        ref_gap_cycles++;
        if (traj_at_goal_snapshot)
        {
            release_active_traj();
            reset_zero_command_memory();
            publish_zero_command();
            RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 1500,
                                 "Trajectory ended at goal, release active reference and hold zero cmd");
            return;
        }
        if (continuous_replan_mode_)
        {
            // In continuous mode, don't stop completely when the trajectory
            // runs out — a new trajectory will arrive shortly.
            // Maintain significant forward speed to avoid stop-start cycling.
            cmd.linear.x = (last_cmd_vel_ >= 0.0) ? 0.70 : -0.25;
            cmd.angular.z = last_cmd_steer_;
            pos_cmd_pub_->publish(cmd);
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1500,
                                 "Trajectory gap in continuous mode, coasting (gap=%d vel=%.2f)",
                                 ref_gap_cycles, cmd.linear.x);
            return;
        }
        // Safety-first behavior: without a valid reference, output zero command.
        reset_zero_command_memory();
        if (traj_at_goal_snapshot)
        {
            planner_traj_valid_ = false;
            try {
                drawRefPath();
            } catch (const std::exception& e) {
                RCLCPP_WARN(node_->get_logger(), "Error keeping terminal reference visualization: %s", e.what());
            }
            publish_zero_command();
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1500,
                                 "Active trajectory exhausted before vehicle reached final pose, hold zero cmd and keep terminal reference until planner replans (gap=%d)",
                                 ref_gap_cycles);
            return;
        }
        clearReferenceVisualization();
        planner_traj_valid_ = false;
        publish_zero_command();
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                             "No reference points available from trajectory, force zero cmd (gap=%d)",
                             ref_gap_cycles);
        return;
    }
    ref_gap_cycles = 0;

    double dist_to_final_snapshot = std::numeric_limits<double>::infinity();
    double yaw_gap_to_final_snapshot = 0.0;
    double terminal_lon_snapshot = 0.0;
    bool terminal_region_snapshot = false;
    bool terminal_overrun_snapshot = false;
    if (has_final_pt_snapshot)
    {
        dist_to_final_snapshot = std::hypot(
            final_pt_snapshot.x - state_snapshot.x,
            final_pt_snapshot.y - state_snapshot.y);
        yaw_gap_to_final_snapshot = final_pt_snapshot.theta - state_snapshot.theta;
        while (yaw_gap_to_final_snapshot > M_PI) yaw_gap_to_final_snapshot -= 2.0 * M_PI;
        while (yaw_gap_to_final_snapshot < -M_PI) yaw_gap_to_final_snapshot += 2.0 * M_PI;
        terminal_lon_snapshot =
            std::cos(final_pt_snapshot.theta) * (state_snapshot.x - final_pt_snapshot.x) +
            std::sin(final_pt_snapshot.theta) * (state_snapshot.y - final_pt_snapshot.y);
        const double time_to_tail = traj_duration_snapshot - ref_tracking_time_snapshot;
        terminal_region_snapshot =
            dist_to_final_snapshot < 1.20 ||
            (traj_duration_snapshot > 1e-3 && time_to_tail < 1.20);
        terminal_overrun_snapshot =
            terminal_region_snapshot &&
            dist_to_final_snapshot < 0.85 &&
            terminal_lon_snapshot > 0.18;
    }

    if (terminal_overrun_snapshot)
    {
        release_active_traj();
        reset_zero_command_memory();
        publish_zero_command();
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1200,
                             "Terminal overrun detected: dist=%.2fm lon=%.2fm yaw_gap=%.2frad, force stop",
                             dist_to_final_snapshot, terminal_lon_snapshot, yaw_gap_to_final_snapshot);
        return;
    }

    if (terminal_region_snapshot && has_final_pt_snapshot)
    {
        const double final_ex_body =
            std::cos(state_snapshot.theta) * (final_pt_snapshot.x - state_snapshot.x) +
            std::sin(state_snapshot.theta) * (final_pt_snapshot.y - state_snapshot.y);
        if (dist_to_final_snapshot < 0.75 && final_ex_body > -0.10)
        {
            final_pt_snapshot.v = 0.0;
            final_pt_snapshot.cur = 0.0;
            std::fill(xref.begin(), xref.end(), final_pt_snapshot);
            smooth_yaw(xref, state_snapshot.theta);
        }
    }
    
    // Publish desired state
    {
        TrajPoint tp = xref.front();
        mpc_controller::msg::DesiredState ds;
        ds.px = tp.x;
        ds.py = tp.y;
        ds.yaw = tp.theta;
        ds.stamp = node_->get_clock()->now();
        Eigen::Vector2d err_vec(tp.x - state_snapshot.x, tp.y - state_snapshot.y);
        ds.tkerror = err_vec.norm();
        double err_yaw = tp.theta - state_snapshot.theta;
        
        if(ds.tkerror > 1.0){
            ds.tkerror = 0.0;
            err_yaw = 0.0;
        }
        else{
            while (err_yaw > M_PI)
                err_yaw -= 2.0 * M_PI;
            while (err_yaw < -M_PI)
                err_yaw += 2.0 * M_PI;
        }
        ds.tkyaw = err_yaw;
        desiredS_pub_->publish(ds);
        
        if(err_vec.norm()<1.0&&err_vec.norm()>0.001){
            errs.push_back(err_vec.norm());
            errs_yaw.push_back(fabs(err_yaw));
            std_msgs::msg::Float64 err_msg;
            err_msg.data = errs.back();
            err_pub->publish(err_msg);
        }
    }
    
    // Check if goal is reached.
    bool physically_reached = false;
    if (has_final_pt_snapshot)
    {
        if ((dist_to_final_snapshot < 0.18 && std::abs(yaw_gap_to_final_snapshot) < 0.65) ||
            (dist_to_final_snapshot < 0.10 && std::abs(yaw_gap_to_final_snapshot) < 1.00)) {
            physically_reached = true;
        }
    }

    if (physically_reached)
    {
        release_active_traj();
        reset_zero_command_memory();
        try {
            drawRefPath();
        } catch (const std::exception& e) {
            RCLCPP_WARN(node_->get_logger(), "Error keeping terminal reference visualization: %s", e.what());
        }

        if (bk_mode)
        {
            Eigen::Vector3d initp = Eigen::Vector3d::Zero();
            double mean_err = 0.0;
            if (!errs.empty()) mean_err = std::accumulate(errs.begin(), errs.end(), 0.0) / (1.0 * errs.size());
            mean_err_all += mean_err;
            outfile << mean_err << std::endl;
            if (initp == Eigen::Vector3d::Zero())
            {
                outfile << "all_mean_track_err: " << mean_err_all <<std::endl;
                outfile.close();
                RCLCPP_WARN(rclcpp::get_logger("mpc"), "all_mean_track_err: %lf", mean_err_all);
                RCLCPP_WARN(rclcpp::get_logger("mpc"), "Benchmark Done.");
                receive_traj = false;
                return;
            }
            geometry_msgs::msg::Point init_point;

            init_point.x = initp.x();
            init_point.y = initp.y();
            init_point.z = initp.z();

            receive_traj = false;
            errs.clear();
            errs_yaw.clear();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            eight_path = traj_analyzer.getTajWps(0.1);
            receive_traj = true;
        }

        double mean_err = 0.0;
        if (!errs.empty()) mean_err = std::accumulate(errs.begin(), errs.end(), 0.0) / (1.0 * errs.size());
        for (size_t i = 0; i < vels.size(); i++)
            vels[i] = fabs(vels[i]);

        if (!has_output)
        {
            std::cout<<"<<<Statistical Datas>>>:"<<std::endl;
            if (!errs.empty()) {
                std::cout <<"max tracking-error: "<<*max_element(errs.begin(),errs.end())<<std::endl;
                std::cout<<"mean tracking-error: "<<mean_err<<std::endl;
            }
            if (!errs_yaw.empty()) {
                std::cout <<"max tkyaw-error: "<<*max_element(errs_yaw.begin(),errs_yaw.end())<<std::endl;
            }
            std::cout<<"trajectory length  : "<<trajectory_length<<std::endl;
            if (!errs.empty()) {
                std::cout<<"end position err   : "<<errs.back()<<std::endl;
            }
            

            TrajPoint p_now = xref.empty() ? TrajPoint() : xref.front();
            double yaw_err = p_now.theta - state_snapshot.theta;
            while (yaw_err > M_PI)
                yaw_err -= 2.0 * M_PI;
            while (yaw_err < -M_PI)
                yaw_err += 2.0 * M_PI;
            std::cout<<"end yaw err        : "<<fabs(yaw_err)<<std::endl;

            errs.clear();
            vels.clear();
            angle_vels.clear();
            errs_yaw.clear();
            trajectory_length = 0.0;
            has_output = true;
        }
    }
    else
    {
        // Degenerate reference check: only skip solver when ALL reference
        // points are at essentially the same position.  Zero reference
        // velocity alone is NOT degenerate — the solver can track position
        // even if ref_v=0 (e.g., approaching a stop point).
        bool ref_degenerate = false;
        if (!xref.empty() && xref.size() >= 2)
        {
            const double dx_ref = xref.back().x - xref.front().x;
            const double dy_ref = xref.back().y - xref.front().y;
            ref_degenerate = (std::abs(dx_ref) < 0.02 && std::abs(dy_ref) < 0.02);
        }
        
        // [修复2] 计算偏差时也要用预测点，防止即将到达时产生急刹震荡
        const double dist_to_ref = xref.empty() ? 0.0 : std::hypot(xref.front().x - pred_x, xref.front().y - pred_y);
        if (ref_degenerate && dist_to_ref < 0.08)
        {
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
        }
        else
        {
            // Fixed: 延迟补偿的预测状态 (pred_x, pred_y, pred_yaw) 已经提前计算，并传递给 getCmd()
            // getCmd() 内部会使用这些预测的状态作为起点，不再重复计算。
            const bool terminal_window =
                has_final_pt_snapshot && xref.size() >= 2 &&
                (terminal_region_snapshot ||
                 std::hypot(xref.back().x - xref.front().x,
                            xref.back().y - xref.front().y) < 0.20);
            getCmd(pred_x, pred_y, pred_yaw, state_snapshot, terminal_window);
        }

        vels.push_back(cmd.linear.x);
        angle_vels.push_back(fabs(cmd.angular.z));
    }
    
    // Always publish the command.  Apply a final wall-clock slew limiter here
    // because sim time can advance in coarse chunks and later post-processing
    // caps can otherwise bypass the limiter inside getCmd().
    const auto wall_now = std::chrono::steady_clock::now();
    const double wall_sec =
        std::chrono::duration<double>(wall_now.time_since_epoch()).count();
    geometry_msgs::msg::Twist cmd_pub = cmd;
    cmd_pub.linear.x = cmd_pub.linear.x * cmd_speed_scale_;
    if (cmd_is_yaw_rate_)
    {
        const double steer = std::clamp(cmd.angular.z, -max_steer, max_steer);
        const double wb = std::max(1e-3, wheel_base);
        const double max_yaw_rate = std::max(0.2, max_speed * std::tan(max_steer) / wb);
        cmd_pub.angular.z = std::clamp(cmd.linear.x * std::tan(steer) / wb, -max_yaw_rate, max_yaw_rate);
    }

    if (last_cmd_publish_wall_sec_ > 0.0)
    {
        const double pub_dt = std::clamp(wall_sec - last_cmd_publish_wall_sec_, 0.01, 0.06);
        const double max_pub_dv = std::max(0.0, max_accel) * pub_dt;
        const double max_pub_ds = std::max(0.0, max_dsteer) * pub_dt;
        cmd_pub.linear.x = std::clamp(cmd_pub.linear.x,
                                      last_cmd_vel_ - max_pub_dv,
                                      last_cmd_vel_ + max_pub_dv);
        cmd_pub.angular.z = std::clamp(cmd_pub.angular.z,
                                       last_cmd_steer_ - max_pub_ds,
                                       last_cmd_steer_ + max_pub_ds);
        if (!cmd_is_yaw_rate_)
        {
            cmd.linear.x = (std::abs(cmd_speed_scale_) > 1e-6)
                ? cmd_pub.linear.x / cmd_speed_scale_
                : cmd_pub.linear.x;
            cmd.angular.z = cmd_pub.angular.z;
            u_0(0, 0) = cmd.linear.x;
            u_0(1, 0) = cmd.angular.z;
            if (U_sol.rows() >= 2 && U_sol.size2() > 0)
            {
                U_sol(0, 0) = cmd.linear.x;
                U_sol(1, 0) = cmd.angular.z;
            }
        }
    }

    // EMA smooth to prevent visible stutter from discrete speed thresholds
    // in the geometric tracking controller.  alpha=0.45 gives ~2-sample
    // smoothing at 50Hz with minimal control lag (~40ms effective delay).
    {
        static double ema_vel = 0.0;
        static double ema_steer = 0.0;
        const double alpha = 0.45;
        ema_vel = alpha * cmd_pub.linear.x + (1.0 - alpha) * ema_vel;
        ema_steer = alpha * cmd_pub.angular.z + (1.0 - alpha) * ema_steer;
        cmd_pub.linear.x = ema_vel;
        cmd_pub.angular.z = ema_steer;
    }

    pos_cmd_pub_->publish(cmd_pub);
    const double prev_published_v = last_cmd_vel_;
    const double prev_published_steer = last_cmd_steer_;
    last_cmd_vel_ = cmd_pub.linear.x;
    last_cmd_steer_ = cmd_pub.angular.z;

    if (cmd_diag_window_start_wall_sec_ <= 0.0)
        cmd_diag_window_start_wall_sec_ = wall_sec;
    if (last_cmd_publish_wall_sec_ > 0.0)
        cmd_diag_max_gap_sec_ = std::max(cmd_diag_max_gap_sec_, wall_sec - last_cmd_publish_wall_sec_);
    last_cmd_publish_wall_sec_ = wall_sec;
    cmd_diag_max_dv_ = std::max(cmd_diag_max_dv_, std::abs(cmd_pub.linear.x - prev_published_v));
    cmd_diag_max_ds_ = std::max(cmd_diag_max_ds_, std::abs(cmd_pub.angular.z - prev_published_steer));
    ++cmd_diag_publish_count_;
    const double diag_window = wall_sec - cmd_diag_window_start_wall_sec_;
    if (diag_window >= 2.0)
    {
        const double hz = static_cast<double>(cmd_diag_publish_count_) / std::max(1e-3, diag_window);
        RCLCPP_INFO(node_->get_logger(),
                    "MPC cmd diag: pub_hz=%.1f max_gap=%.0fms max_dv=%.3f max_dsteer=%.3f cmd=(%.3f, %.3f)",
                    hz, cmd_diag_max_gap_sec_ * 1000.0, cmd_diag_max_dv_, cmd_diag_max_ds_,
                    cmd_pub.linear.x, cmd_pub.angular.z);
        cmd_diag_window_start_wall_sec_ = wall_sec;
        cmd_diag_max_gap_sec_ = 0.0;
        cmd_diag_max_dv_ = 0.0;
        cmd_diag_max_ds_ = 0.0;
        cmd_diag_publish_count_ = 0;
    }

    if (control_visualization_period_sec_ > 1e-3 &&
        (last_control_vis_pub_sec_ <= 0.0 ||
         now_sec < last_control_vis_pub_sec_ ||
         now_sec - last_control_vis_pub_sec_ >= control_visualization_period_sec_))
    {
        try {
            drawFollowPath();
            drawVehicleHistory();
            drawRefPath();
            drawPredictPath();
            last_control_vis_pub_sec_ = now_sec;
            last_follow_vis_pub_sec_ = now_sec;
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                 "MPC visualization skipped: %s", e.what());
        }
    }

    static int print_count = 0;
    if ((print_count++ % 60) == 0) {
        RCLCPP_INFO(node_->get_logger(), "MPC status: has_odom=%d receive_traj=%d xref=%zu at_goal=%d cmd=(%.3f, %.3f)",
                    (int)has_odom_snapshot, (int)receive_traj_snapshot, xref.size(), (int)traj_at_goal_snapshot,
                    cmd.linear.x, cmd.angular.z);
    }
}

// Fixed: 将 getCmd 内部的延迟补偿逻辑删除，因为它已经被提升到 cmdCallback 作用域的顶部
// 现在的 getCmd 只需要接收 cmdCallback 传递过来的 pred_x, pred_y, pred_yaw

void MPC::buildNLP()
{
    // Build the CasADi Opti problem ONCE.  The reference trajectory and
    // initial state/control are parameters that get updated each cycle.
    casadi::Dict options;
    casadi::Dict qpsol_options;
    options["print_status"] = false;
    options["print_time"] = false;
    options["print_header"] = false;
    options["print_iteration"] = false;
    options["verbose"] = false;
    options["verbose_init"] = false;
    options["max_iter"] = 120;
    options["tol_pr"] = 1e-3;
    options["tol_du"] = 5e-4;
    options["qpsol"] = "qrqp";
    options["error_on_fail"] = false;
    qpsol_options["print_header"] = false;
    qpsol_options["print_iter"] = false;
    qpsol_options["error_on_fail"] = false;
    options["qpsol_options"] = qpsol_options;

    nlp = casadi::Opti();

    X = nlp.variable(3, Npre);
    U = nlp.variable(2, Npre);

    // Parameters: updated every cycle
    ref_x_sym_     = nlp.parameter(Npre, 1);
    ref_y_sym_     = nlp.parameter(Npre, 1);
    ref_theta_sym_ = nlp.parameter(Npre, 1);
    ref_v_sym_     = nlp.parameter(Npre, 1);
    ref_cur_sym_   = nlp.parameter(Npre, 1);
    X0_param_      = nlp.parameter(3, 1);
    U0_param_      = nlp.parameter(2, 1);

    J = 0;
    Slice all;

    for (int i = 0; i < Npre; i++)
    {
        MX ref_x     = ref_x_sym_(i);
        MX ref_y     = ref_y_sym_(i);
        MX ref_theta = ref_theta_sym_(i);
        MX ref_state = vertcat(ref_x, ref_y, ref_theta);
        
        // Keep the optimizer state-tracking dominated.  DP velocity/curvature
        // can be noisy after frequent replans; treating them as hard control
        // references makes the solver chase feedforward instead of recovering
        // lateral/yaw error.
        MX ref_v     = 0;
        MX ref_steer = 0;
        MX ref_u     = vertcat(ref_v, ref_steer);

        MX x_next;
        if (i == 0)
            x_next = stateTrans(X0_param_, U(all, i));
        else
            x_next = stateTrans(X(all, i - 1), U(all, i));

        nlp.subject_to(X(all, i) == x_next);
        nlp.subject_to(U_min <= U(all, i) <= U_max);

        // [修正] 恢复线性角度误差: 你的代码中 smooth_yaw() 已经在外部完美解卷了 2PI。
        // 移除非线性的 sin(e/2)，因为它会破坏 SQP 二次规划子问题的凸性，导致高速下求解器崩溃(Max_Iter_Exceeded)。
        MX dx = ref_state(0) - X(0, i);
        MX dy = ref_state(1) - X(1, i);
        MX dtheta = ref_state(2) - X(2, i); 
        MX delta_x = vertcat(dx, dy, dtheta);
        
        MX delta_u = ref_u - U(all, i);

        MX du;
        if (i > 0)
            du = U(all, i) - U(all, i - 1);
        else
            du = U(all, i) - U0_param_;

        J = J + mtimes(delta_x.T(), mtimes(Q, delta_x));
        J = J + mtimes(delta_u.T(), mtimes(R, delta_u));
        J = J + mtimes(du.T(), mtimes(Rd, du));
    }

    nlp.minimize(J);
    nlp.solver("sqpmethod", options);

    // Initialize solver buffers
    X_sol = DM::zeros(3, Npre);
    U_sol = DM::zeros(2, Npre);
    u_0 = DM::zeros(2, 1);

    nlp_built_ = true;
    RCLCPP_INFO(node_->get_logger(), "MPC NLP built once (Npre=%d), will reuse every cycle", Npre);
}

void MPC::getCmd(double pred_x, double pred_y, double pred_yaw,
                 const MPCState &state_snapshot, bool terminal_window)
{
    const double prev_v = cmd.linear.x;
    const double prev_steer = cmd.angular.z;
    MPCState pred_state = state_snapshot;
    pred_state.x = pred_x;
    pred_state.y = pred_y;
    pred_state.theta = pred_yaw;

    auto wrap_angle = [](double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    };

    struct TrackingInfo
    {
        double dist = 0.0;
        double lat = 0.0;       // positive: vehicle is left of the reference tangent
        double lon = 0.0;
        double yaw_err = 0.0;   // reference yaw minus vehicle yaw
    };

    auto tracking_info = [&](const MPCState &s) {
        TrackingInfo info;
        if (xref.empty())
            return info;

        const TrajPoint &r0 = xref.front();
        const double dx = s.x - r0.x;
        const double dy = s.y - r0.y;
        const double cr = std::cos(r0.theta);
        const double sr = std::sin(r0.theta);
        info.dist = std::hypot(dx, dy);
        info.lon = cr * dx + sr * dy;
        info.lat = -sr * dx + cr * dy;
        info.yaw_err = wrap_angle(r0.theta - s.theta);
        return info;
    };

    auto tracking_speed_cap = [&](const TrackingInfo &info) {
        double cap = max_speed;
        auto soft_cap = [&](double value, double begin, double end, double min_cap) {
            if (value <= begin)
                return max_speed;
            const double ratio = std::clamp((value - begin) / std::max(1e-3, end - begin), 0.0, 1.0);
            const double blend = ratio * ratio * (3.0 - 2.0 * ratio);
            return max_speed + blend * (std::min(min_cap, max_speed) - max_speed);
        };

        cap = std::min(cap, soft_cap(info.dist, 0.55, 1.85, 1.30));
        cap = std::min(cap, soft_cap(std::abs(info.lat), 0.40, 1.15, 1.20));
        cap = std::min(cap, soft_cap(std::abs(info.yaw_err), 0.70, 1.70, 0.90));
        if (terminal_window)
        {
            const double terminal_cap = (info.dist < 0.18)
                                            ? 0.20
                                            : std::clamp(0.22 + 0.75 * info.dist, 0.42, 0.90);
            cap = std::min(cap, terminal_cap);
        }
        return std::max(0.0, cap);
    };

    auto geometric_tracking_cmd = [&](bool solver_failed) {
        Eigen::Vector2d out(0.0, 0.0);
        if (xref.empty())
            return out;

        const TrackingInfo info = tracking_info(pred_state);
        const bool ref_reverse = std::isfinite(xref.front().v) && xref.front().v < -0.03;
        double nominal_speed = std::isfinite(xref.front().v) ? std::abs(xref.front().v) : 0.0;
        const double preview_span = std::hypot(xref.back().x - pred_state.x,
                                               xref.back().y - pred_state.y);

        if (nominal_speed < 0.12)
        {
            const TrajPoint &tail = xref[std::min<int>(static_cast<int>(xref.size()) - 1, std::max(1, Npre / 2))];
            const double span = std::hypot(tail.x - pred_state.x, tail.y - pred_state.y);
            nominal_speed = (terminal_window && span < 0.22) ? 0.0 : 1.30;
        }
        if (!terminal_window)
        {
            const double cruise_floor = (preview_span > 1.65) ? 1.75 :
                                        (preview_span > 1.05) ? 1.45 :
                                        (preview_span > 0.50) ? 0.85 : 0.35;
            if (info.dist < 1.00 && std::abs(info.yaw_err) < 1.00)
                nominal_speed = std::max(nominal_speed, cruise_floor);
            else
                nominal_speed = std::max(nominal_speed, 0.90);
        }

        double cap = tracking_speed_cap(info);
        if (solver_failed)
            cap = std::min(cap, 0.90);

        double v_abs = std::min(nominal_speed, cap);
        if (!terminal_window && v_abs < 0.30)
            v_abs = 0.30;
        if (terminal_window && info.dist < 0.12)
            v_abs = 0.0;

        bool has_forward_ref = false;
        for (const auto &p : xref)
        {
            const double dx = p.x - pred_state.x;
            const double dy = p.y - pred_state.y;
            const double ex_body = std::cos(pred_state.theta) * dx + std::sin(pred_state.theta) * dy;
            if (ex_body > 0.15)
            {
                has_forward_ref = true;
                break;
            }
        }
        if (!ref_reverse && !has_forward_ref && info.dist > 0.50)
            v_abs = terminal_window ? 0.0 : std::min(v_abs, 0.45);

        const double v_cmd = ref_reverse
            ? -std::min(v_abs, reverse_speed_cap)
            : std::min(v_abs, max_speed);

        const double lookahead = terminal_window
            ? std::clamp(0.55 + 0.25 * std::abs(v_cmd), 0.50, 1.00)
            : std::clamp(0.75 + 0.45 * std::abs(v_cmd), 0.75, 1.75);
        int target_idx = static_cast<int>(xref.size()) - 1;
        for (int i = 0; i < static_cast<int>(xref.size()); ++i)
        {
            const double d = std::hypot(xref[i].x - pred_state.x, xref[i].y - pred_state.y);
            if (d >= lookahead)
            {
                target_idx = i;
                break;
            }
        }

        const double tx = xref[target_idx].x - pred_state.x;
        const double ty = xref[target_idx].y - pred_state.y;
        const double target_dist = std::max(0.45, std::hypot(tx, ty));
        double alpha = wrap_angle(std::atan2(ty, tx) - pred_state.theta);
        if (ref_reverse)
            alpha = wrap_angle(alpha + M_PI);

        const double steer_pp = std::atan2(2.0 * wheel_base * std::sin(alpha), target_dist);
        const double cur_limit = std::tan(max_steer) / std::max(1e-3, wheel_base);
        const double target_cur = std::isfinite(xref[target_idx].cur)
            ? std::clamp(xref[target_idx].cur, -cur_limit, cur_limit)
            : 0.0;
        const double steer_ff = std::atan(wheel_base * target_cur);
        const double speed_for_gain = std::max(0.80, std::abs(v_cmd));
        const double lat_gain = terminal_window ? 0.45 : 0.34;
        const double yaw_gain = terminal_window ? 0.45 : 0.32;
        const double pp_gain = terminal_window ? 0.60 : 0.42;
        const double ff_gain = terminal_window ? 0.20 : 0.35;
        const double lat_term = -std::atan2(lat_gain * info.lat, speed_for_gain + 0.45);
        const double yaw_term = yaw_gain * info.yaw_err;
        const double steer_cmd = std::clamp(pp_gain * steer_pp + ff_gain * steer_ff + lat_term + yaw_term,
                                           -max_steer, max_steer);

        out.x() = std::clamp(v_cmd, std::max(min_speed, -reverse_speed_cap), max_speed);
        out.y() = steer_cmd;
        return out;
    };

    if (U_sol.rows() != 2 || static_cast<int>(U_sol.size2()) != Npre ||
        X_sol.rows() != 3 || static_cast<int>(X_sol.size2()) != Npre)
    {
        X_sol = DM::zeros(3, Npre);
        U_sol = DM::zeros(2, Npre);
    }

    bool used_nmpc_solution = false;

    if (!use_nmpc_solver_)
    {
        const Eigen::Vector2d tracking_cmd = geometric_tracking_cmd(false);
        cmd.linear.x = tracking_cmd.x();
        cmd.angular.z = tracking_cmd.y();

        for (int i = 0; i < Npre; i++)
        {
            U_sol(0, i) = cmd.linear.x;
            U_sol(1, i) = cmd.angular.z;
        }
    }
    else
    {
    if (!casadi_param_inited_)
    {
        Q = DM::zeros(3, 3);
        Q(0, 0) = q_diag_[0];
        Q(1, 1) = q_diag_[1];
        Q(2, 2) = q_diag_[2];

        R = DM::zeros(2, 2);
        R(0, 0) = r_diag_[0];
        R(1, 1) = r_diag_[1];

        Rd = DM::zeros(2, 2);
        Rd(0, 0) = rd_diag_[0];
        Rd(1, 1) = rd_diag_[1];

        U_min = DM::zeros(2, 1);
        U_max = DM::zeros(2, 1);
        U_min(0, 0) = std::max(min_speed, -reverse_speed_cap);
        U_min(1, 0) = -max_steer;
        U_max(0, 0) = max_speed;
        U_max(1, 0) = max_steer;
        casadi_param_inited_ = true;
    }

    // ── NMPC: CasADi SQP solver (build once, reuse per cycle) ──
    if (!nlp_built_)
        buildNLP();

    // Pack reference into parameter vectors
    std::vector<double> rx(Npre), ry(Npre), rt(Npre), rv(Npre), rc(Npre);
    for (int i = 0; i < Npre; i++)
    {
        const int ri = std::min(i, static_cast<int>(xref.size()) - 1);
        rx[i] = xref[ri].x;  ry[i] = xref[ri].y;  rt[i] = xref[ri].theta;
        rv[i] = xref[ri].v;  rc[i] = xref[ri].cur;
    }
    nlp.set_value(ref_x_sym_, DM(rx));     nlp.set_value(ref_y_sym_, DM(ry));
    nlp.set_value(ref_theta_sym_, DM(rt)); nlp.set_value(ref_v_sym_, DM(rv));
    nlp.set_value(ref_cur_sym_, DM(rc));

    // Fixed: getCmd 接收 cmdCallback 传递过来的 pred_x, pred_y, pred_yaw
    x_0 = {pred_x, pred_y, pred_yaw};
    nlp.set_value(X0_param_, x_0);
    nlp.set_value(U0_param_, u_0);

    // Seed an initial guess that satisfies the dynamics constraint.  Use a
    // geometry-based tracking seed rather than DP curvature feedforward, so a
    // noisy curvature field or a large replan jump cannot warm-start the SQP in
    // the wrong steering direction.
    if (!xref.empty())
    {
        const Eigen::Vector2d seed_cmd = geometric_tracking_cmd(false);
        for (int i = 0; i < Npre; i++)
        {
            const int ri = std::min(i, (int)xref.size() - 1);
            double ref_v = std::isfinite(xref[ri].v) ? xref[ri].v : 0.0;
            if (std::abs(ref_v) < 0.05 || std::signbit(ref_v) != std::signbit(seed_cmd.x()))
                ref_v = seed_cmd.x();
            U_sol(0, i) = std::clamp(ref_v, std::max(min_speed, -reverse_speed_cap), max_speed);
            U_sol(1, i) = std::clamp(seed_cmd.y(), -max_steer, max_steer);
        }

        // Forward-propagate state from current vehicle state using control guess.
        double cx = pred_x, cy = pred_y, cyaw = pred_yaw; // Fixed: 使用传递进来的预测状态作为初始 guess
        for (int i = 0; i < Npre; i++)
        {
            double v = static_cast<double>(U_sol(0, i));
            double steer = static_cast<double>(U_sol(1, i));
            cx += dt * std::cos(cyaw) * v;
            cy += dt * std::sin(cyaw) * v;
            cyaw += dt * v * std::tan(steer) / wheel_base;
            X_sol(0, i) = cx;
            X_sol(1, i) = cy;
            X_sol(2, i) = cyaw;
        }
    }
    nlp.set_initial(X, X_sol);
    nlp.set_initial(U, U_sol);

    try
    {
        const casadi::OptiSol sol = nlp.solve();
        X_sol = sol.value(X);  U_sol = sol.value(U);
        Slice all;  DM cmd_0 = U_sol(all, 0);
        u_0 = cmd_0;
        cmd.linear.x = (double)cmd_0(0, 0);
        cmd.angular.z = (double)cmd_0(1, 0);
        used_nmpc_solution = true;
    }
    catch(const std::exception& e)
    {
        static int fail_n = 0;  fail_n++;
        if (fail_n % 50 == 1)
            RCLCPP_ERROR(node_->get_logger(), "MPC fail(%d): %s | v=%.2f",
                        fail_n, e.what(), prev_v);
        const Eigen::Vector2d fallback_cmd = geometric_tracking_cmd(true);
        cmd.linear.x = fallback_cmd.x();
        cmd.angular.z = fallback_cmd.y();

        // Reset warm-start with forward-propagated feasible initial guess
        const Eigen::Vector2d seed_cmd = geometric_tracking_cmd(true);
        for (int i = 0; i < Npre; i++)
        {
            const int ri = xref.empty() ? 0 : std::min(i, (int)xref.size() - 1);
            double ref_v = xref.empty() ? seed_cmd.x() : xref[ri].v;
            if (!std::isfinite(ref_v) || std::abs(ref_v) < 0.05 ||
                std::signbit(ref_v) != std::signbit(seed_cmd.x()))
                ref_v = seed_cmd.x();
            U_sol(0, i) = std::clamp(ref_v, std::max(min_speed, -reverse_speed_cap), max_speed);
            U_sol(1, i) = std::clamp(seed_cmd.y(), -max_steer, max_steer);
        }
        {
            double cx = pred_x, cy = pred_y, cyaw = pred_yaw; // Fixed: 使用传递进来的预测状态作为初始 guess
            for (int i = 0; i < Npre; i++)
            {
                double v = static_cast<double>(U_sol(0, i));
                double steer = static_cast<double>(U_sol(1, i));
                cx += dt * std::cos(cyaw) * v;
                cy += dt * std::sin(cyaw) * v;
                cyaw += dt * v * std::tan(steer) / wheel_base;
                X_sol(0, i) = cx;
                X_sol(1, i) = cy;
                X_sol(2, i) = cyaw;
            }
        }
    }
    }

    if (!std::isfinite(cmd.linear.x) || !std::isfinite(cmd.angular.z))
    {
        const Eigen::Vector2d fallback_cmd = geometric_tracking_cmd(true);
        cmd.linear.x = fallback_cmd.x();
        cmd.angular.z = fallback_cmd.y();
    }

    // Enforce command bounds and slew-rate limits to prevent spinning behavior.
    cmd.linear.x = std::clamp(cmd.linear.x, std::max(min_speed, -reverse_speed_cap), max_speed);
    cmd.angular.z = std::clamp(cmd.angular.z, -max_steer, max_steer);
    const double cmd_now_sec = node_->get_clock()->now().seconds();
    double cmd_dt = dt;
    if (last_cmd_callback_time_sec_ > 0.0 && cmd_now_sec >= last_cmd_callback_time_sec_)
        cmd_dt = std::clamp(cmd_now_sec - last_cmd_callback_time_sec_, 0.02, 0.12);
    last_cmd_callback_time_sec_ = cmd_now_sec;
    const double max_dv = std::max(0.0, max_accel) * cmd_dt;
    const double max_ds = std::max(0.0, max_dsteer) * cmd_dt;
    cmd.linear.x = std::clamp(cmd.linear.x, prev_v - max_dv, prev_v + max_dv);
    cmd.angular.z = std::clamp(cmd.angular.z, prev_steer - max_ds, prev_steer + max_ds);

    if (!xref.empty())
    {
        const double dx_ref = xref.front().x - state_snapshot.x;
        const double dy_ref = xref.front().y - state_snapshot.y;
        const double dist_ref = std::hypot(dx_ref, dy_ref);
        const double ex_body = std::cos(state_snapshot.theta) * dx_ref + std::sin(state_snapshot.theta) * dy_ref;
        double dyaw_ref = xref.front().theta - state_snapshot.theta;
        while (dyaw_ref > M_PI) dyaw_ref -= 2.0 * M_PI;
        while (dyaw_ref < -M_PI) dyaw_ref += 2.0 * M_PI;

        const double now_sec = node_->get_clock()->now().seconds();
        const bool ref_speed_reverse = (xref.front().v < -0.03);
        const bool ref_prefers_reverse = ref_speed_reverse;
        const bool request_reverse = (cmd.linear.x < -0.02) || ref_prefers_reverse;
        const bool allow_reverse_enter = ref_prefers_reverse;
        const bool should_exit_reverse = !ref_prefers_reverse &&
                                         ((std::abs(dyaw_ref) < 0.60) || (ex_body > 0.22));

        if (ref_prefers_reverse && now_sec >= reverse_hold_until_sec_)
        {
            reverse_allowed_ = true;
            reverse_hold_until_sec_ = now_sec + reverse_switch_cooldown_sec_;
        }

        if (request_reverse != reverse_allowed_ && now_sec >= reverse_hold_until_sec_)
        {
            if (request_reverse && allow_reverse_enter)
            {
                reverse_allowed_ = true;
                reverse_hold_until_sec_ = now_sec + reverse_switch_cooldown_sec_;
            }
            if (!request_reverse && should_exit_reverse)
            {
                reverse_allowed_ = false;
                reverse_hold_until_sec_ = now_sec + reverse_switch_cooldown_sec_;
            }
        }

        double reverse_follow_dist = dist_ref;
        if (reverse_follow_dist <= 0.35 && xref.size() >= 3)
        {
            const TrajPoint &far_ref = xref.back();
            reverse_follow_dist = std::hypot(far_ref.x - state_snapshot.x, far_ref.y - state_snapshot.y);
        }
        if (ref_speed_reverse && reverse_follow_dist > 0.40 && cmd.linear.x > -reverse_follow_speed_floor_)
        {
            const double reverse_floor = std::min(reverse_follow_speed_floor_, std::max(0.08, reverse_speed_cap * 0.85));
            cmd.linear.x = -reverse_floor;
        }

        if (!reverse_allowed_ && cmd.linear.x < 0.0)
        {
            double forward_dist = dist_ref;
            if (forward_dist <= 0.40 && xref.size() >= 3)
            {
                const TrajPoint &far_ref = xref.back();
                forward_dist = std::hypot(far_ref.x - state_snapshot.x, far_ref.y - state_snapshot.y);
            }
            cmd.linear.x = (forward_dist > 0.40) ? 0.06 : 0.0;
        }

        if (enable_aggressive_recovery_)
        {
            // Optional startup boost for specific scenes; disabled by default
            // because it can mask planning/controller synchronization issues.
            const double since_accept = (node_->get_clock()->now() - planner_traj_last_accept_stamp_).seconds();
            double boost_dist = dist_ref;
            if (boost_dist <= 0.55 && xref.size() >= 3)
            {
                const TrajPoint &far_ref = xref.back();
                boost_dist = std::hypot(far_ref.x - state_snapshot.x, far_ref.y - state_snapshot.y);
            }
            if (since_accept >= 0.0 && since_accept < startup_crawl_boost_sec_ && boost_dist > 0.90 && std::abs(cmd.linear.x) < 0.06)
            {
                const double v_sign = reverse_allowed_ ? -1.0 : 1.0;
                cmd.linear.x = 0.14 * v_sign;
            }
        }
    }

    if (!xref.empty() && cmd.linear.x < 0.0)
    {
        const double dx_ref = xref.front().x - state_snapshot.x;
        const double dy_ref = xref.front().y - state_snapshot.y;
        const double ex_body = std::cos(state_snapshot.theta) * dx_ref + std::sin(state_snapshot.theta) * dy_ref;
        double dyaw_ref = xref.front().theta - state_snapshot.theta;
        while (dyaw_ref > M_PI) dyaw_ref -= 2.0 * M_PI;
        while (dyaw_ref < -M_PI) dyaw_ref += 2.0 * M_PI;
        // If near-term reference is straight ahead, avoid reverse chattering.
        const bool ref_speed_reverse = (xref.front().v < -0.03);
        if (!ref_speed_reverse && ex_body > 0.25 && std::abs(dyaw_ref) < 0.60)
        {
            cmd.linear.x = 0.05;
        }
    }

    if (!xref.empty())
    {
        const TrackingInfo info = tracking_info(state_snapshot);
        const double cap = tracking_speed_cap(info);
        if (std::abs(cmd.linear.x) > cap)
            cmd.linear.x = std::copysign(cap, cmd.linear.x);
    }

    // ── Minimal post-processing: only physical safety limits ──
    // The SQP solver is trusted to compute optimal commands.  We only apply
    // hard physical constraints that the solver cannot enforce internally.

    // Curvature speed cap: physical lateral acceleration limit.
    if (!xref.empty())
    {
        const double ref_cur = std::abs(xref.front().cur);
        if (std::isfinite(ref_cur) && ref_cur > 1e-3)
        {
            const double lat_acc_limit = 2.50;
            const double min_curve_cap = terminal_window ? 0.25 : std::min(turn_speed_min_cap_, max_speed);
            const double v_cap = std::clamp(std::sqrt(lat_acc_limit / std::max(1e-3, ref_cur)), min_curve_cap, max_speed);
            if (cmd.linear.x >= 0.0)
                cmd.linear.x = std::min(cmd.linear.x, v_cap);
            else
                cmd.linear.x = std::max(cmd.linear.x, -std::min(reverse_speed_cap, v_cap * 0.85));
        }
    }

    // If steering flips sign at high speed, gently damp velocity to prevent
    // lateral oscillation.  Use a softer cap (0.6 m/s instead of 0.25) to
    // avoid abrupt control discontinuity that disrupts tracking.
    if (used_nmpc_solution && (prev_steer * cmd.angular.z) < -0.08 && std::abs(prev_v) > 0.8)
        cmd.linear.x = std::copysign(std::min(std::abs(cmd.linear.x), 0.60), cmd.linear.x);

    // Absolute bounds.
    cmd.linear.x = std::clamp(cmd.linear.x, std::max(min_speed, -reverse_speed_cap), max_speed);
    cmd.angular.z = std::clamp(cmd.angular.z, -max_steer, max_steer);

    if (U_sol.rows() >= 2 && U_sol.size2() > 0)
    {
        double last_v = cmd.linear.x;
        double last_steer = cmd.angular.z;
        double cx = pred_x;
        double cy = pred_y;
        double cyaw = pred_yaw;
        for (int i = 0; i < Npre && i < static_cast<int>(U_sol.size2()); i++)
        {
            double v = (i == 0) ? cmd.linear.x : static_cast<double>(U_sol(0, i));
            double steer = (i == 0) ? cmd.angular.z : static_cast<double>(U_sol(1, i));
            v = std::clamp(v, std::max(min_speed, -reverse_speed_cap), max_speed);
            steer = std::clamp(steer, -max_steer, max_steer);
            if (i > 0)
            {
                v = std::clamp(v, last_v - max_dv, last_v + max_dv);
                steer = std::clamp(steer, last_steer - max_ds, last_steer + max_ds);
            }
            U_sol(0, i) = v;
            U_sol(1, i) = steer;
            last_v = v;
            last_steer = steer;
            cx += dt * std::cos(cyaw) * v;
            cy += dt * std::sin(cyaw) * v;
            cyaw += dt * v * std::tan(steer) / wheel_base;
            X_sol(0, i) = cx;
            X_sol(1, i) = cy;
            X_sol(2, i) = cyaw;
        }
    }

    u_0 = DM::zeros(2, 1);
    u_0(0, 0) = cmd.linear.x;
    u_0(1, 0) = cmd.angular.z;
    
    if (delay_num>0)
    {
        output_buff.erase(output_buff.begin());
        output_buff.push_back(Eigen::Vector2d(cmd.linear.x, cmd.angular.z));
    }
}
