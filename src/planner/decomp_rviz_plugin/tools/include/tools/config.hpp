#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>
#include <memory>
#include <Eigen/Eigen>
#include <rclcpp/rclcpp.hpp>
struct Config
{
    double mini_T = 0.0;
    double wei_time_ = 500.0;
    double kmax = 0.45;
    double vmax = 2.0;
    double latAccmax = 3.0;
    double lonAccmax = 3.0;
    double accRatemax = 8.0;
    double kdotmax = 5.0;
    double yawdotmax = 3.14;
    int traj_res = 16;
    double scaling_wf_min_ = 0.01, scaling_wc_min_ = 0.01;
    double rho = 1.0, rho_max =1000.0;
    double gamma = 1;
    double cons_eps_ = 0.20;
    int mem_size = 256;
    int past = 3; //3 
    double g_epsilon = 1.0e-3;
    double min_step = 1.0e-32;
    double delta = 1.0e-4;
    int max_iterations = 5000;
    int amlMaxIter = 15;
    double pieceTime = 0.7;
    std::vector<Eigen::Vector2d> conpts_array;
    std::vector<double> start_vec;
    std::vector<double> end_vec;
    Eigen::Vector4d start_state; 
    Eigen::Vector4d end_state; 
    double FIRI_bound;
    std::vector<Eigen::Vector2d> conpts; 
    /*map parameters*/
    double mapRes = 0.1;
    double mapX = 50.0, mapY = 50.0, mapZ = 10.0;
    int expandSize = 1;
    /*front end*/
    double wheel_base = 0.6;
    double horizon_ = 50.0;
    double yaw_resolution_ = 0.3;
    double lambda_heu_ = 5.0;
    int allocate_num_ = 100000;
    int check_num_ = 5;
    double max_seach_time = 1.0;
    double step_arc = 0.9;
    double checkl = 0.2;
    double non_siguav = 0.0;
    double backW = 2.0;
    double penaWei = 500.0;
    double esdfWei = 1000.0;
    double backend_curvature_weight = 0.0;
    double backend_curvature_violation_weight = 0.0;
    double backend_curvature_soft_ratio = 0.85;
    double backend_lateral_acc_weight = 0.0;
    double backend_jerk_weight = 0.1;
    double miniS =0.9;
    bool isdebug = true;
    bool isfixGear = false;
    bool isVis = false;
    bool enable_shot = false;
    double safeMargin = 0.1;
    double riskWei = 0.85;
    double risk_safe_threshold = 0.75;
    double risk_obstacle_threshold = 0.85;
    double risk_free_threshold = 0.40;
    double phimax = 0.785;
    double omegamax = 0.3;
    double steer_res = 0.5;
    bool once_vis_one = false;
    int once_duration_ms = 500;
    int skip_iter_duration_ms, skip_iter_vis;
    bool once_optimization;
    bool individual_vis = true;
    bool front_end_vis = true;


    Config(std::shared_ptr<rclcpp::Node> node)
    {
        // ROS2 parameter access using declare_parameter and get_parameter
        mini_T = node->declare_parameter<double>("mini_T", 0.0);
        wei_time_ = node->declare_parameter<double>("wei_time_", 500.0);
        kmax = node->declare_parameter<double>("kmax", 0.45);
        vmax = node->declare_parameter<double>("vmax", 3.0);
        latAccmax = node->declare_parameter<double>("latAccmax", 3.0);
        lonAccmax = node->declare_parameter<double>("lonAccmax", 3.0);
        accRatemax = node->declare_parameter<double>("accRatemax", 8.0);
        kdotmax = node->declare_parameter<double>("kdotmax", 1.0);
        yawdotmax = node->declare_parameter<double>("yawdotmax", 3.14);
        traj_res = node->declare_parameter<int>("traj_res", 16);
        rho = node->declare_parameter<double>("rho", 1.0);
        rho_max = node->declare_parameter<double>("rho_max", 1000.0);
        gamma = node->declare_parameter<double>("gamma", 1.0);
        cons_eps_ = node->declare_parameter<double>("cons_eps_", 0.2);
        mem_size = node->declare_parameter<int>("mem_size", 256);
        past = node->declare_parameter<int>("past", 3);
        g_epsilon = node->declare_parameter<double>("g_epsilon", 1.0e-3);
        min_step = node->declare_parameter<double>("min_step", 1.0e-32);
        delta = node->declare_parameter<double>("delta", 1.0e-4);
        max_iterations = node->declare_parameter<int>("max_iterations", 5000);
        amlMaxIter = node->declare_parameter<int>("amlMaxIter", 15);
        pieceTime = node->declare_parameter<double>("pieceTime", 0.7);
        FIRI_bound = node->declare_parameter<double>("FIRI_bound", 3.0);
        once_vis_one = node->declare_parameter<bool>("once_vis_one", false);
        once_optimization = node->declare_parameter<bool>("once_optimization", false);
        once_duration_ms = node->declare_parameter<int>("once_duration_ms", 500);
        skip_iter_vis = node->declare_parameter<int>("skip_iter_vis", 10);
        skip_iter_duration_ms = node->declare_parameter<int>("skip_iter_duration_ms", 500);
        front_end_vis = node->declare_parameter<bool>("front_end_vis", false);
        individual_vis = node->declare_parameter<bool>("individual_vis", true);
        penaWei = node->declare_parameter<double>("penaWei", 500.0);
        esdfWei = node->declare_parameter<double>("esdfWei", 1000.0);
        riskWei = node->declare_parameter<double>("riskWei", 0.85);
        risk_safe_threshold = node->declare_parameter<double>("risk_safe_threshold", 0.75);
        risk_obstacle_threshold = node->declare_parameter<double>("risk_obstacle_threshold", 0.85);
        risk_free_threshold = node->declare_parameter<double>("risk_free_threshold", 0.40);
        backend_curvature_weight = node->declare_parameter<double>("backend/curvature_weight", 0.0);
        backend_curvature_violation_weight =
            node->declare_parameter<double>("backend/curvature_violation_weight", 0.0);
        backend_curvature_soft_ratio =
            node->declare_parameter<double>("backend/curvature_soft_ratio", 0.85);
        backend_lateral_acc_weight = node->declare_parameter<double>("backend/lateral_acc_weight", 0.0);
        backend_jerk_weight = node->declare_parameter<double>("backend/jerk_weight", 0.1);
        miniS = node->declare_parameter<double>("miniS", 0.9);
        if (!node->has_parameter("safeMargin")) safeMargin = node->declare_parameter<double>("safeMargin", 0.1);
        else safeMargin = node->get_parameter("safeMargin").as_double();
        phimax = node->declare_parameter<double>("phimax", 0.785);
        omegamax = node->declare_parameter<double>("omegamax", 0.3);
        steer_res = node->declare_parameter<double>("steer_res", 0.5);

        std::vector<double> conpts_flat_default{
            0.31,  0.29,
            0.31, -0.29,
           -0.31, -0.29,
           -0.31,  0.29
        };
        std::vector<double> conpts_flat =
            node->declare_parameter<std::vector<double>>("conpts_flat", conpts_flat_default);

        // Backward compatibility: allow a flat `conpts` double-array in ROS2 params.
        if (node->has_parameter("conpts"))
        {
            try
            {
                auto legacy = node->get_parameter("conpts");
                if (legacy.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY)
                {
                    auto legacy_flat = legacy.as_double_array();
                    if (!legacy_flat.empty())
                    {
                        conpts_flat.assign(legacy_flat.begin(), legacy_flat.end());
                    }
                }
            }
            catch (...)
            {
            }
        }

        if (conpts_flat.size() < 6 || (conpts_flat.size() % 2) != 0)
        {
            RCLCPP_WARN(node->get_logger(),
                        "Invalid conpts_flat size=%zu, fallback to default rectangular footprint",
                        conpts_flat.size());
            conpts_flat = conpts_flat_default;
        }
        conpts.clear();
        conpts.reserve(conpts_flat.size() / 2);
        for (size_t i = 0; i + 1 < conpts_flat.size(); i += 2)
        {
            conpts.emplace_back(conpts_flat[i], conpts_flat[i + 1]);
        }

        start_vec = node->declare_parameter<std::vector<double>>("start_vec", std::vector<double>{0, 0, 0, 0});
        end_vec = node->declare_parameter<std::vector<double>>("end_vec", std::vector<double>{1, 1, 0, 0});
        if(start_vec.size() >= 4) {
            start_state = Eigen::Vector4d{start_vec[0], start_vec[1], start_vec[2], start_vec[3]};
        }
        if(end_vec.size() >= 4) {
            end_state = Eigen::Vector4d{end_vec[0], end_vec[1], end_vec[2], end_vec[3]};
        }
        
        mapRes = node->declare_parameter<double>("mapRes", 0.1);
        mapX = node->declare_parameter<double>("mapX", 50.0);
        mapY = node->declare_parameter<double>("mapY", 50.0);
        mapZ = node->declare_parameter<double>("mapZ", 10.0);
        expandSize = node->declare_parameter<int>("expandSize", 2);
        wheel_base = node->declare_parameter<double>("wheel_base", 0.6);
        horizon_ = node->declare_parameter<double>("horizon_", 50.0);
        yaw_resolution_ = node->declare_parameter<double>("yaw_resolution_", 0.3);
        lambda_heu_ = node->declare_parameter<double>("lambda_heu_", 5.0);
        allocate_num_ = node->declare_parameter<int>("allocate_num_", 100000);
        check_num_ = node->declare_parameter<int>("check_num_", 5);
        max_seach_time = node->declare_parameter<double>("max_seach_time", 1.0);
        step_arc = node->declare_parameter<double>("step_arc", 0.9);
        checkl = node->declare_parameter<double>("checkl", 0.2);
        non_siguav = node->declare_parameter<double>("non_siguav", 0.0);
        isdebug = node->declare_parameter<bool>("isdebug", false);
        isfixGear = node->declare_parameter<bool>("isfixGear", false);
        isVis = node->declare_parameter<bool>("isVis", false);
        enable_shot = node->declare_parameter<bool>("enable_shot", false);
        backW = node->declare_parameter<double>("backW", 2.0);
    }
};
inline double normalize_angle(const double& theta) {
    double theta_tmp = theta;    
    while(theta_tmp >= acos(-1.0)){
        theta_tmp -= 2 * acos(-1.0);
    }
    while(theta_tmp < -acos(-1.0)){
        theta_tmp += 2 * acos(-1.0);
    } 
  return theta_tmp;
}
typedef std::shared_ptr<Config> ConfigPtr;

#endif 
