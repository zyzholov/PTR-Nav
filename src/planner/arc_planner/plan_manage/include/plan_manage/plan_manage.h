#ifndef PLAN_MANAGE_HPP
#define PLAN_MANAGE_HPP
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <mutex>
#include <iostream>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <path_searching/kino_astar.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <mpc_controller/msg/se2_traj.hpp>
#include <mpc_controller/msg/poly_traj.hpp>
#include <mpc_controller/msg/single_poly.hpp>
#include <mpc_controller/msg/single_poly_ac.hpp>
#include <mpc_controller/msg/poly_traj_ac.hpp>
#include <mpc_controller/msg/d_ptraj_container.hpp>

namespace visualization {
class Visualization;
}

#include <arcPlan/Trajopt_alm.hpp>

#define BUDGET 0.1
namespace plan_manage{
    class PlanManager{
        public:
            PlanManager(){};
            void init(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<se2_grid::SE2Grid> se2_grid_map, std::shared_ptr<se2_grid::SE2Grid> sdf_grid_map);
            void updateOdom(double x, double y, double yaw){
                odom << x, y, yaw;
                hasOdom = true;
            }
            void setTarget(double x, double y, double yaw){
                targetPose << x, y, yaw;
                hasTarget = true;
            }
            bool process();
            bool replanLightweight();  // skip Kino-A*, extend current traj, reduced optimization
            void clearVisualizationPath();
            void setParkingEnd(Eigen::Vector4d end_pt);
            void setTrajStartTime(const double& traj_start_time);
            double getTrajStartTime();
            void setInitStateAndInput(const Eigen::Vector4d& state, const double& start_time);
            void publishSimTraj2Controller();
            double getDuration();
            bool getPredictState(const double pre_t, Eigen::Vector4d& state);
            void republishTrajectory();  // bump start_time, same path — no mutation
        private:
            rclcpp::Publisher<mpc_controller::msg::DPtrajContainer>::SharedPtr mpc_polynome_pub_;
            bool hasTarget = false, hasOdom = false;
            Eigen::Vector3d targetPose, odom;
            std::shared_ptr<visualization::Visualization> vis_tool;
            std::shared_ptr<Config> config_;
            std::unique_ptr<path_searching::KinoAstar> kino_path_finder_;
            std::shared_ptr<se2_grid::SE2Grid> se2_grid_map_;
            std::shared_ptr<se2_grid::SE2Grid> sdf_grid_map_;
            rclcpp::Time begin_time;
            Eigen::Matrix<double, 2, 3> iniState2d, finState2d;
            Eigen::Vector4d start_state_, end_state_;
            Eigen::Vector2d start_ctrl_;
            double start_time_;
            Eigen::Vector4d new_start_state_;
            double new_start_time_;
            Eigen::MatrixXd initInnerPts2d;
            void warm();
            double pieceTime;
            /*ros related*/
            rclcpp::TimerBase::SharedPtr processTimer, cudaWarmTimer;
            rclcpp::Subscription<mpc_controller::msg::DPtrajContainer>::SharedPtr targetSub;
            rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub;
            rclcpp::Publisher<mpc_controller::msg::DPtrajContainer>::SharedPtr trajCmdPub;
            rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr visPathPub;
            rclcpp::TimerBase::SharedPtr visPathKeepaliveTimer;
            rclcpp::TimerBase::SharedPtr traj_republish_timer_;
            double traj_republish_hz_ = 10.0;
            std::shared_ptr<rclcpp::Node> node_;
            mpc_controller::msg::DPtrajContainer trajmsg;
            PolyTrajOpt::UgvTrajectory optTraj;
            nav_msgs::msg::Path lastVisPath;
            bool hasVisPath = false;
            bool publish_optimized_path_visualization_ = true;
            bool publish_rough_preview_ = false;
            bool publish_local_horizon_ = false;
            double local_publish_horizon_sec_ = 6.0;
            double traj_publish_min_sdf_ = 0.20;
            double traj_publish_sdf_tolerance_ = 0.004;
            double traj_publish_risk_hard_ = 0.84;
            bool traj_publish_require_valid_layer_ = false;
            double traj_publish_max_start_gap_ = 0.35;
            double traj_publish_max_start_yaw_gap_ = 0.75;
            double traj_safety_ignore_start_sec_ = 0.60;
            double traj_safety_ignore_start_dist_ = 0.35;
            double traj_safety_check_horizon_sec_ = 6.0;
            double traj_publish_max_duration_ = 4.0;
            bool traj_publish_allow_partial_on_gate_ = true;
            double traj_publish_partial_min_safe_sec_ = 1.0;
            double traj_publish_partial_backoff_sec_ = 0.35;
            bool traj_publish_partial_active_ = false;
            double traj_publish_partial_duration_ = -1.0;
            double published_traj_duration_ = -1.0;
            double partial_frontier_goal_gap_thresh_ = 1.6;
            double partial_frontier_max_length_ = 12.0;
            int max_piece_num_per_seg_ = 6;
            int max_piece_num_per_seg_partial_ = 4;
            int rt_max_iterations_cap_ = 450;
            int rt_aml_max_iter_cap_ = 10;
            int rough_path_smoothing_passes_ = 2;
            double rough_path_smoothing_alpha_ = 0.32;
            double rough_path_smoothing_max_deviation_ = 0.28;
            double rough_path_smoothing_risk_slack_ = 0.03;
            bool front_end_allow_reverse_ = false;

            void republishVisualizationPath();
            bool validateTrajectorySafety(double &min_sdf, double &max_risk);
            bool validateTrajectorySafety(const PolyTrajOpt::UgvTrajectory &traj,
                                          double &min_sdf, double &max_risk);

            // ego_planner-style: reuse previous path if still valid
            std::vector<Eigen::Vector3d> prev_samples_;
            Eigen::Vector4d prev_end_state_ = Eigen::Vector4d::Zero();
            bool has_prev_samples_ = false;

            // Thread safety: background thread writes optTraj (process/replanLightweight),
            // main thread reads optTraj (getPredictState, publishSimTraj2Controller, etc.)
            mutable std::mutex traj_mutex_;
            mutable std::mutex viz_mutex_;
    };
}

#endif
