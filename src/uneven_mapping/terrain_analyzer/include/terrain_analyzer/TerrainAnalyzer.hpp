#pragma once

#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/se2_grid_ros.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <message_filters/cache.h>
#include <message_filters/subscriber.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/thread.hpp>

#include "terrain_analyzer/sensor_processors/sensor_processors.hpp"
#include "terrain_analyzer/post_processors/post_processors.hpp"
#include "terrain_analyzer/utils/utils.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace terrain_analyzer
{
    struct ClampVar
    {
        ClampVar(const float& minVar, const float& maxVar) : minVar_(minVar), maxVar_(maxVar) {}
        const float operator()(const float& x) const
        {
            return x < minVar_ ? minVar_ : (x > maxVar_ ? std::numeric_limits<float>::infinity() : x);
        }
        float minVar_, maxVar_;
    };

    class TerrainAnalyzer
    {
        public:
#define ENABLE_CUDA 1
#ifdef ENABLE_CUDA
            TerrainAnalyzer() : 
                fused_map({"elevation", "var", "inpainted", "smooth", "normal_x", "normal_y", "risk", "zbx", "zby"}, 
                          {false, false, false, false, false, false, true, true, true}),
                sdf_map({"sdf"}, {false}),
                post_processors("se2_grid::SE2Grid") {}
#else
            TerrainAnalyzer() : 
                raw_map({"elevation", "var", "var_x", "var_y", "var_xy"}, 
                        {false, false, false, false, false}),
                fused_map({"elevation", "upper_bound", "lower_bound"},
                        {false, false, false}),
                post_processors("se2_grid::SE2Grid")
            {
                raw_basic_layers.clear();
                fused_basic_layers.clear();
                raw_basic_layers.emplace_back(std::string("elevation"));
                raw_basic_layers.emplace_back(std::string("var"));
                fused_basic_layers.emplace_back(std::string("elevation"));
                fused_basic_layers.emplace_back(std::string("lower_bound"));
                fused_basic_layers.emplace_back(std::string("upper_bound"));
            }
#endif

            bool init(ros::NodeHandle& nh);
            bool add(const pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudMap, Eigen::VectorXf& variances);
            bool fuseAll();

            void mapCallback(const ros::TimerEvent& /*event*/);
            void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg); 

            inline float cumulativeDistributionFunction(float x, float mean, float standardDeviation)
            {
                return 0.5 * erfc(-(x - mean) / (standardDeviation * sqrt(2.0)));
            }
            
        private:
            // params
            int cache_size = 200;
            bool verbose = false;
            double min_var;
            double max_var;
            double min_var_hori;
            double max_var_hori;
            double mahalanobis_threshold;
            sensor_msgs::PointCloud2 global_cloud_msg;

            // data
            message_filters::Cache<nav_msgs::Odometry> odom_cache;
            OusterLidarProcessor sensor_processor;
            PostProcessorChain<se2_grid::SE2Grid> post_processors;
            se2_grid::SE2Grid raw_map;
            se2_grid::SE2Grid fused_map;
            se2_grid::SE2Grid sdf_map;
            std::vector<std::string> raw_basic_layers;
            std::vector<std::string> fused_basic_layers;

            // ros
            message_filters::Subscriber<nav_msgs::Odometry> odom_sub;
            ros::Publisher  raw_pub;
            ros::Publisher  fused_pub;
            ros::Publisher  sdf_pub;
            ros::Publisher  normal_pub;
            ros::Publisher  shit_pub;
            ros::Subscriber cloud_sub;
            ros::Timer      map_timer;
            ros::Time       last_update_time;
    };
}