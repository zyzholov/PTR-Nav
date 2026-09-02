#pragma once

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <numeric>
#include <memory>
#include <chrono>
#include <thread>
#include <limits>
#include <atomic>
#include <mutex>

#include <casadi/casadi.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
// #include "kinematics_simulator/ControlCmd.h"
#include "utils/traj_anal.hpp"

#include <mpc_controller/msg/se2_traj.hpp>
#include <mpc_controller/msg/poly_traj.hpp>
#include <mpc_controller/msg/single_poly.hpp>
#include <mpc_controller/msg/d_ptraj_container.hpp>
#include <mpc_controller/msg/desired_state.hpp>

using namespace std;
using namespace Eigen;
using namespace casadi;

class MPCState
{
public:
    double x = 0;
    double y = 0;
    double z = 0;
    double theta = 0;
};

class MPC
{
private:
    // parameters
    /// algorithm param
    double du_th = 0.1;
    double dt = 0.2;
    int Npre = 5;
    int delay_num;

    /// constraints
    double max_steer = M_PI / 4;
    double max_dsteer = M_PI / 6;
    double max_speed = 55.0 / 3.6;
    double min_speed = -55.0 / 3.6;
    double reverse_speed_cap = 0.6;
    double max_accel = 1.0; 
    double wheel_base = 0.5; 

    // MPC dataset
    casadi::Opti nlp;
    casadi::DM X_sol;
    casadi::DM U_sol;
    casadi::DM x_0; // Initial state
    casadi::DM u_0; // Initial input

    casadi::MX J;
    casadi::MX X;
    casadi::MX U;

    casadi::DM Q;
    casadi::DM R;
    casadi::DM Rd;

    casadi::DM U_min;
    casadi::DM U_max;
    bool solver_buf_inited_ = false;
    bool casadi_param_inited_ = false;

    // Pre-built NLP: CasADi Opti is constructed ONCE and reused every cycle.
    // Rebuilding it each time (original behavior) triggers codegen/JIT overhead.
    bool nlp_built_ = false;
    casadi::MX ref_x_sym_, ref_y_sym_, ref_theta_sym_, ref_v_sym_, ref_cur_sym_;
    casadi::MX X0_param_, U0_param_;
    void buildNLP();
    std::vector<double> q_diag_;
    std::vector<double> r_diag_;
    std::vector<double> rd_diag_;
    std::vector<TrajPoint> xref;
    std::vector<geometry_msgs::msg::Point> vehicle_history;  // Track vehicle trajectory
    std::vector<geometry_msgs::msg::Point> mission_history;  // Track trajectory for current mission
    std::vector<geometry_msgs::msg::Point> latched_ref_path; // Keep full planned path visible during mission
    size_t history_max = 500;  // Keep last 500 points
    size_t mission_history_max = 2000;
    std::vector<Eigen::Vector2d> output_buff;
    std::atomic_bool cmd_running_{false};
    std::mutex odom_mutex_;
    std::mutex traj_mutex_;
    std::shared_ptr<rclcpp::Node> node_shared_;

    // control data
    bool has_odom;
    bool receive_traj = false;
    MPCState now_state;
    TrajAnalyzer traj_analyzer;
    TrajAnalyzer new_traj_analyzer;
    bool planner_traj_valid_ = false;
    rclcpp::Time planner_traj_stamp_;
    double fallback_block_after_planner_sec_ = 2.0;
    bool enable_fallback_goal_ = false;
    bool cmd_is_yaw_rate_ = false;
    bool use_nmpc_solver_ = true;
    rclcpp::Time planner_traj_last_accept_stamp_;
    double planner_update_min_interval_sec_ = 1.2;
    double planner_update_max_jump_m_ = 2.0;
    double replan_cautious_sec_ = 1.0;
    double replan_cautious_speed_cap_ = 0.14;
    double replan_cautious_dist_thresh_ = 0.50;
    double replan_cautious_yaw_thresh_ = 0.55;
    double replan_cautious_steer_thresh_ = 0.24;
    bool reverse_allowed_ = false;
    double reverse_hold_until_sec_ = 0.0;
    double reverse_switch_cooldown_sec_ = 0.8;
    double cmd_speed_scale_ = 1.0;
    double turn_speed_steer_gain_ = 0.55;
    double turn_speed_min_cap_ = 0.12;
    bool force_reverse_follow_ = true;
    double reverse_follow_speed_floor_ = 0.12;
    double reverse_hint_hold_sec_ = 3.0;
    bool traj_prefers_reverse_ = false;
    double traj_reverse_ratio_ = 0.0;
    double traj_reverse_hint_until_sec_ = 0.0;
    double startup_crawl_boost_sec_ = 1.2;
    bool enable_aggressive_recovery_ = false;
    bool ref_tracking_valid_ = false;
    double ref_tracking_time_ = 0.0;
    bool continuous_replan_mode_ = false;
    double last_cmd_vel_ = 0.0;
    double last_cmd_steer_ = 0.0;
    double last_cmd_callback_time_sec_ = 0.0;
    double last_cmd_publish_wall_sec_ = 0.0;
    double cmd_diag_window_start_wall_sec_ = 0.0;
    double cmd_diag_max_gap_sec_ = 0.0;
    double cmd_diag_max_dv_ = 0.0;
    double cmd_diag_max_ds_ = 0.0;
    int cmd_diag_publish_count_ = 0;
    double last_ref_projection_sec_ = 0.0;
    double last_follow_vis_pub_sec_ = 0.0;
    double last_control_vis_pub_sec_ = 0.0;
    double ref_projection_period_sec_ = 0.08;
    double control_visualization_period_sec_ = 0.50;

    // ros interface
    rclcpp::Node* node_ = nullptr;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
    rclcpp::CallbackGroup::SharedPtr odom_cb_group_;
    rclcpp::CallbackGroup::SharedPtr traj_cb_group_;
    rclcpp::TimerBase::SharedPtr cmd_timer_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pos_cmd_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr history_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_pub, predict_pub, ref_pub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr err_pub;
    rclcpp::Publisher<mpc_controller::msg::DesiredState>::SharedPtr desiredS_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<mpc_controller::msg::SE2Traj>::SharedPtr traj_sub_1;
    rclcpp::Subscription<mpc_controller::msg::PolyTraj>::SharedPtr traj_sub_2;
    rclcpp::Subscription<mpc_controller::msg::DPtrajContainer>::SharedPtr traj_sub_3;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr trigger_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr trigger_sub_legacy_;
    geometry_msgs::msg::Twist cmd;
    void cmdCallback();
    void rcvOdomCallBack(const nav_msgs::msg::Odometry::SharedPtr msg);
    void rcvTrajCallBack(const mpc_controller::msg::SE2Traj::SharedPtr msg);
    void rcvFlatTrajCallBack(const mpc_controller::msg::PolyTraj::SharedPtr msg);
    void rcvDPTrajCallBack(const mpc_controller::msg::DPtrajContainer::SharedPtr msg);
    void rcvTriggerCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    // for test tracking performance
    bool test_mpc;
    vector<TrajPoint> eight_path;
    
    // for benchmark
    bool bk_mode;
    bool has_output = false;
    double mean_err_all = 0.0;
    double trajectory_length = 0.0;
    double mean_acc = 0.0;
    double mean_steer_vel = 0.0;
    double mean_jerk = 0.0;
    std::vector<double> errs;
    std::vector<double> errs_yaw;
    std::vector<double> errs_lon;
    std::vector<double> errs_lat;
    std::vector<double> vels;
    std::vector<double> angle_vels;
    std::vector<double> jerks;
    string traj_file;
    std::ofstream outfile;

    // MPC function
    void getCmd(double pred_x, double pred_y, double pred_yaw, const MPCState &state_snapshot, bool terminal_window);

    void normlize_theta(double& th)
    {
        while (th > M_PI)
            th -= M_PI * 2;
        while (th < -M_PI)
            th += M_PI * 2;
    }

    void smooth_yaw(vector<TrajPoint> &ref_points, double current_theta)
    {
        if (ref_points.empty())
            return;

        double dyaw = ref_points[0].theta - current_theta;

        while (dyaw >= M_PI / 2)
        {
            ref_points[0].theta -= M_PI * 2;
            dyaw = ref_points[0].theta - current_theta;
        }
        while (dyaw <= -M_PI / 2)
        {
            ref_points[0].theta += M_PI * 2;
            dyaw = ref_points[0].theta - current_theta;
        }

        for (size_t i = 0; i + 1 < ref_points.size(); i++)
        {
            dyaw = ref_points[i + 1].theta - ref_points[i].theta;
            while (dyaw >= M_PI / 2)
            {
                ref_points[i + 1].theta -= M_PI * 2;
                dyaw = ref_points[i + 1].theta - ref_points[i].theta;
            }
            while (dyaw <= -M_PI / 2)
            {
                ref_points[i + 1].theta += M_PI * 2;
                dyaw = ref_points[i + 1].theta - ref_points[i].theta;
            }
        }

        return;
    }

    MX stateTrans(MX x_now, MX u_now)
    {
        MX x_next;
        // Keep steer away from tan(pi/2) singularity for numerical stability.
        const double steer_limit = std::min(max_steer, 1.2);
        MX steer_safe = fmax(fmin(u_now(1), steer_limit), -steer_limit);
        MX x_next_x = dt * mtimes(cos(x_now(2)), u_now(0)) + x_now(0);
        MX x_next_y = dt * mtimes(sin(x_now(2)), u_now(0)) + x_now(1);
        MX x_next_theta = dt * u_now(0) * tan(steer_safe) / wheel_base + x_now(2);
        x_next = vertcat(x_next_x, x_next_y, x_next_theta);

        return x_next;
    }

    void drawFollowPath(void)
    {
        if (mission_history.empty()) return;

        visualization_msgs::msg::MarkerArray arr;
        auto now = node_->get_clock()->now();

        visualization_msgs::msg::Marker clear;
        clear.header.frame_id = "map";
        clear.header.stamp = now;
        clear.ns = "__following_clear__";
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        clear.id = 0;
        arr.markers.push_back(clear);

        visualization_msgs::msg::Marker glow;
        glow.header.frame_id = "map";
        glow.header.stamp = now;
        glow.ns = "following_glow";
        glow.type = visualization_msgs::msg::Marker::LINE_STRIP;
        glow.action = visualization_msgs::msg::Marker::ADD;
        glow.id = 0;
        glow.pose.orientation.w = 1.0;
        glow.color.r = 0.10;
        glow.color.g = 0.85;
        glow.color.b = 1.0;
        glow.color.a = 0.22;
        glow.scale.x = 0.78;
        glow.points = mission_history;
        arr.markers.push_back(glow);

        visualization_msgs::msg::Marker core;
        core.header.frame_id = "map";
        core.header.stamp = now;
        core.ns = "following_core";
        core.type = visualization_msgs::msg::Marker::LINE_STRIP;
        core.action = visualization_msgs::msg::Marker::ADD;
        core.id = 1;
        core.pose.orientation.w = 1.0;
        core.color.r = 0.05;
        core.color.g = 0.95;
        core.color.b = 1.0;
        core.color.a = 0.58;
        core.scale.x = 0.42;
        core.points = mission_history;
        arr.markers.push_back(core);

        vis_pub->publish(arr);
    }

    void drawPredictPath(void)
    {
        if (X_sol.rows() == 0 || X_sol.size2() == 0) return;

        // Collect points
        std::vector<geometry_msgs::msg::Point> pts;
        geometry_msgs::msg::Point pt;
        Slice all;
        for (int i = 0; i < Npre && i < (int)X_sol.size2(); i++)
        {
            DM pre_state = X_sol(all, i);
            pt.x = double(pre_state(0, 0));
            pt.y = double(pre_state(1, 0));
            pt.z = now_state.z + 0.02;
            pts.push_back(pt);
        }
        if (pts.size() < 2) return;

        visualization_msgs::msg::MarkerArray arr;
        auto now = node_->get_clock()->now();
        visualization_msgs::msg::Marker clear;
        clear.header.frame_id = "map";
        clear.header.stamp = now;
        clear.ns = "__predict_clear__";
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        clear.id = 0;
        arr.markers.push_back(clear);

        visualization_msgs::msg::Marker glow;
        glow.header.frame_id = "map";
        glow.header.stamp = now;
        glow.ns = "predict_glow";
        glow.type = visualization_msgs::msg::Marker::LINE_STRIP;
        glow.action = visualization_msgs::msg::Marker::ADD;
        glow.id = 0;
        glow.pose.orientation.w = 1.0;
        glow.color.r = 0.0;
        glow.color.g = 1.0;
        glow.color.b = 0.35;
        glow.color.a = 0.36;
        glow.scale.x = 0.42;
        glow.points = pts;
        arr.markers.push_back(glow);

        visualization_msgs::msg::Marker core;
        core.header.frame_id = "map";
        core.header.stamp = now;
        core.ns = "predict_core";
        core.type = visualization_msgs::msg::Marker::LINE_STRIP;
        core.action = visualization_msgs::msg::Marker::ADD;
        core.id = 1;
        core.pose.orientation.w = 1.0;
        core.color.r = 0.05;
        core.color.g = 1.0;
        core.color.b = 0.15;
        core.color.a = 1.0;
        core.scale.x = 0.18;
        core.points = pts;
        arr.markers.push_back(core);

        predict_pub->publish(arr);
    }

    void drawVehicleHistory(void)
    {
        if (vehicle_history.empty()) return;
        
        visualization_msgs::msg::Marker line_strip;
        line_strip.header.frame_id = "map";
        line_strip.header.stamp = node_->get_clock()->now();
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_strip.action = visualization_msgs::msg::Marker::ADD;
        line_strip.id = 5000;
        
        line_strip.pose.orientation.w = 1.0;
        line_strip.color.r = 0.04;
        line_strip.color.g = 0.92;
        line_strip.color.b = 1.0;
        line_strip.color.a = 0.46;
        line_strip.scale.x = 0.68;
        
        line_strip.points = vehicle_history;
        history_pub->publish(line_strip);
    }

    std::vector<geometry_msgs::msg::Point> makeSlidingRefPath(void)
    {
        std::vector<geometry_msgs::msg::Point> pts;
        std::vector<TrajPoint> future;
        if (ref_tracking_valid_ && traj_analyzer.usesDpTrajectory())
        {
            future = traj_analyzer.getFutureWpsFromTime(0.1, 10.0, ref_tracking_time_);
        }
        else
        {
            future = traj_analyzer.getFutureWps(0.1, 10.0);
        }
        if (!future.empty())
        {
            pts.reserve(future.size());
            for (const auto &p : future)
            {
                geometry_msgs::msg::Point pt;
                pt.x = p.x;
                pt.y = p.y;
                pt.z = now_state.z + 0.05;
                pts.push_back(pt);
            }
        }
        else
        {
            geometry_msgs::msg::Point pt;
            for (size_t i = 0; i < xref.size(); i++)
            {
                pt.x = xref[i].x;
                pt.y = xref[i].y;
                pt.z = now_state.z + 0.02;
                pts.push_back(pt);
            }
        }
        return pts;
    }

    void drawRefPath(void)
    {
        if (latched_ref_path.empty() && xref.empty()) return;

        std::vector<geometry_msgs::msg::Point> pts = makeSlidingRefPath();
        if (pts.size() < 2) return;

        visualization_msgs::msg::MarkerArray arr;
        auto now = node_->get_clock()->now();
        visualization_msgs::msg::Marker clear;
        clear.header.frame_id = "map";
        clear.header.stamp = now;
        clear.ns = "__reference_clear__";
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        clear.id = 0;
        arr.markers.push_back(clear);

        // Outer glow layer (wide, semi-transparent warm tone)
        visualization_msgs::msg::Marker glow;
        glow.header.frame_id = "map";
        glow.header.stamp = now;
        glow.ns = "reference_glow";
        glow.type = visualization_msgs::msg::Marker::LINE_STRIP;
        glow.action = visualization_msgs::msg::Marker::ADD;
        glow.id = 0;
        glow.pose.orientation.w = 1.0;
        glow.color.r = 1.0;
        glow.color.g = 0.25;
        glow.color.b = 0.55;
        glow.color.a = 0.30;
        glow.scale.x = 0.48;
        glow.points = pts;
        arr.markers.push_back(glow);

        visualization_msgs::msg::Marker core;
        core.header.frame_id = "map";
        core.header.stamp = now;
        core.ns = "reference_core";
        core.type = visualization_msgs::msg::Marker::LINE_STRIP;
        core.action = visualization_msgs::msg::Marker::ADD;
        core.id = 1;
        core.pose.orientation.w = 1.0;
        core.color.r = 1.0;
        core.color.g = 0.18;
        core.color.b = 0.48;
        core.color.a = 0.95;
        core.scale.x = 0.22;
        core.points = pts;
        arr.markers.push_back(core);

        ref_pub->publish(arr);
    }

    void clearVisualization()
    {
        auto now = node_->get_clock()->now();
        auto clear_marker = [](const std::string &frame_id, const rclcpp::Time &stamp) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = stamp;
            marker.ns = "__mpc_clear__";
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker.id = 0;
            marker.pose.orientation.w = 1.0;
            return marker;
        };

        visualization_msgs::msg::MarkerArray clear_arr;
        clear_arr.markers.push_back(clear_marker("map", now));
        vis_pub->publish(clear_arr);
        predict_pub->publish(clear_arr);
        ref_pub->publish(clear_arr);

        auto msg = clear_marker("map", now);
        history_pub->publish(msg);

        vehicle_history.clear();
        mission_history.clear();
        latched_ref_path.clear();
        xref.clear();
    }

    void clearReferenceVisualization()
    {
        auto now = node_->get_clock()->now();
        auto make_clear = [&](visualization_msgs::msg::MarkerArray &arr) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = now;
            marker.ns = "__mpc_clear__";
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            marker.id = 0;
            arr.markers.push_back(marker);
        };

        if (predict_pub)
        {
            visualization_msgs::msg::MarkerArray arr;
            make_clear(arr);
            predict_pub->publish(arr);
        }
        if (ref_pub)
        {
            visualization_msgs::msg::MarkerArray arr;
            make_clear(arr);
            ref_pub->publish(arr);
        }
    }

public:
	MPC() {
    node_ = nullptr;
        cmd_timer_.reset();
        pos_cmd_pub_.reset();
        vis_pub.reset();
        predict_pub.reset();
        ref_pub.reset();
        history_pub.reset();
        err_pub.reset();
        desiredS_pub_.reset();
        odom_sub_.reset();
        traj_sub_1.reset();
        traj_sub_2.reset();
        traj_sub_3.reset();
        trigger_sub_.reset();
        trigger_sub_legacy_.reset();
    }
    void init(std::shared_ptr<rclcpp::Node> node);
	~MPC() {}
};
