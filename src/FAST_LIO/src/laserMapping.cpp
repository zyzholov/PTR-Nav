// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <algorithm>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <limits>
#include <deque>
#include <unordered_map>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "dynamic_voxel_ray.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
string map_frame = "camera_init", body_frame = "body";
vector<double> publish_body_T_param(3, 0.0);
vector<double> publish_body_R_param{1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0};
V3D Publish_T_wrt_IMU(Zero3d);
M3D Publish_R_wrt_IMU(Eye3d);

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool   local_map_pub_en = false;
bool    is_first_lidar = true;
double local_map_radius = 12.0, local_map_z_half = 3.0;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
double accumulated_map_publish_hz = 1.0;
double accumulated_map_voxel_size = 0.30;
double accumulated_map_effective_voxel_size = 0.30;
int accumulated_map_max_points = 800000;
bool accumulated_map_dense = false;
bool dynamic_map_clear_enabled = false;
bool dynamic_map_expand_enabled = false;
int dynamic_map_clear_frames = 3;
int dynamic_map_clear_max_gap_frames = 3;
bool dynamic_map_ground_replace_enabled = false;
int dynamic_map_ground_clear_frames = 3;
int dynamic_map_ground_min_returns = 2;
double dynamic_map_ground_max_span = 0.14;
double dynamic_map_ground_search_radius = 0.24;
double dynamic_map_ground_min_height = 0.30;
int dynamic_map_ground_max_points = 6000;
int dynamic_map_expand_min_seeds = 6;
int dynamic_map_max_rays = 2500;
double dynamic_map_ray_range = 30.0;
double dynamic_map_endpoint_margin = 0.65;
uint32_t accumulated_frame = 0;

struct AccumulatedVoxel
{
    PointType point;
    uint32_t hit_frame = 0;
    uint32_t ground_free_frame = 0;
    dynamic_voxel_ray::EvidenceWindow free_evidence;
    uint8_t ground_free_scans = 0;
};

using AccumulatedVoxelKey = dynamic_voxel_ray::Key;
std::unordered_map<AccumulatedVoxelKey, AccumulatedVoxel, dynamic_voxel_ray::Hash> accumulated_map_voxels;
std::deque<BoxPointType> pending_dynamic_deletions;
struct DynamicClearSeed
{
    AccumulatedVoxelKey key;
    uint32_t frame;
};
std::deque<DynamicClearSeed> dynamic_clear_seeds;
uint64_t dynamic_direct_cleared_total = 0;
uint64_t dynamic_ground_cleared_total = 0;

AccumulatedVoxelKey accumulated_voxel_key(const PointType &point, double voxel_size)
{
    const double safe_voxel_size = std::max(0.05, voxel_size);
    return {
        static_cast<int64_t>(std::floor(static_cast<double>(point.x) / safe_voxel_size)),
        static_cast<int64_t>(std::floor(static_cast<double>(point.y) / safe_voxel_size)),
        static_cast<int64_t>(std::floor(static_cast<double>(point.z) / safe_voxel_size)),
    };
}

void insert_accumulated_point(const PointType &point)
{
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) return;
    auto [it, inserted] = accumulated_map_voxels.try_emplace(
        accumulated_voxel_key(point, accumulated_map_effective_voxel_size), AccumulatedVoxel{point});
    it->second.hit_frame = accumulated_frame;
    it->second.free_evidence.reset();
    it->second.ground_free_scans = 0;
}

void rebuild_accumulated_voxels(double voxel_size)
{
    PointVector points;
    points.reserve(accumulated_map_voxels.size());
    for (const auto &entry : accumulated_map_voxels) points.push_back(entry.second.point);

    accumulated_map_effective_voxel_size = std::max(0.05, voxel_size);
    accumulated_map_voxels.clear();
    dynamic_clear_seeds.clear();
    accumulated_map_voxels.reserve(points.size());
    for (const auto &point : points) insert_accumulated_point(point);
}

void enforce_accumulated_map_budget()
{
    const size_t max_points = static_cast<size_t>(std::max(0, accumulated_map_max_points));
    if (max_points == 0 || accumulated_map_voxels.size() <= max_points) return;

    double new_voxel_size = accumulated_map_effective_voxel_size;
    for (int attempt = 0; attempt < 8 && accumulated_map_voxels.size() > max_points; ++attempt)
    {
        new_voxel_size *= 1.25;
        rebuild_accumulated_voxels(new_voxel_size);
    }
    std::cout << "[FAST_LIO] accumulated map reached its point budget; "
              << "publication voxel size increased to " << accumulated_map_effective_voxel_size
              << " m (" << accumulated_map_voxels.size() << " points)" << std::endl;
}

void refresh_accumulated_map_cloud()
{
    pcl_wait_pub->clear();
    pcl_wait_pub->reserve(accumulated_map_voxels.size());
    for (const auto &entry : accumulated_map_voxels) pcl_wait_pub->push_back(entry.second.point);
    pcl_wait_pub->width = static_cast<uint32_t>(pcl_wait_pub->size());
    pcl_wait_pub->height = 1;
    pcl_wait_pub->is_dense = true;
}

void accumulate_map_frame()
{
    if (feats_down_world->empty()) return;
    ++accumulated_frame;
    if (accumulated_map_dense)
    {
        // Keep publication detail independent of the EKF/IKD registration
        // downsampling. These are measured, deskewed returns, not interpolated
        // points; the accumulated voxel budget still bounds memory use.
        for (const auto &body_point : feats_undistort->points)
        {
            PointType point;
            pointBodyToWorld(&body_point, &point);
            insert_accumulated_point(point);
        }
    }
    else
    {
        for (const auto &point : feats_down_world->points) insert_accumulated_point(point);
    }

    if (dynamic_map_clear_enabled && dynamic_map_max_rays > 0)
    {
        struct GroundSpan { float low; float high; };
        // Preserve the Isaac ground-support scale when publication voxels
        // are finer than registration voxels.
        const double coarse_size = std::max(0.40, 2.0 * accumulated_map_effective_voxel_size);
        std::unordered_map<AccumulatedVoxelKey, GroundSpan, dynamic_voxel_ray::Hash> ground_cells;
        ground_cells.reserve(feats_undistort->size());
        for (const auto &body_point : feats_undistort->points)
        {
            PointType point;
            pointBodyToWorld(&body_point, &point);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
            const AccumulatedVoxelKey key{
                static_cast<int64_t>(std::floor(point.x / coarse_size)),
                static_cast<int64_t>(std::floor(point.y / coarse_size)), 0};
            auto [it, inserted] = ground_cells.try_emplace(key, GroundSpan{point.z, point.z});
            if (!inserted)
            {
                it->second.low = std::min(it->second.low, point.z);
                it->second.high = std::max(it->second.high, point.z);
            }
        }
        const auto supported_elevated = [&](double x, double y, double z) {
            const int64_t coarse_x = static_cast<int64_t>(std::floor(x / coarse_size));
            const int64_t coarse_y = static_cast<int64_t>(std::floor(y / coarse_size));
            unsigned int quadrants = 0;
            for (int gy = -2; gy <= 2; ++gy)
            {
                for (int gx = -2; gx <= 2; ++gx)
                {
                    const auto observed = ground_cells.find({coarse_x + gx, coarse_y + gy, 0});
                    if (observed == ground_cells.end() ||
                        observed->second.high - observed->second.low > 0.20f ||
                        z - observed->second.low < dynamic_map_ground_min_height) continue;
                    quadrants |= 1u << ((gx >= 0 ? 1 : 0) + (gy >= 0 ? 2 : 0));
                }
            }
            return __builtin_popcount(quadrants) >= 3;
        };
        const std::array<double, 3> origin = {pos_lid.x(), pos_lid.y(), pos_lid.z()};
        const size_t stride = std::max<size_t>(1, (feats_down_world->size() + dynamic_map_max_rays - 1) /
                                                   dynamic_map_max_rays);
        std::vector<AccumulatedVoxelKey> erased;
        for (size_t i = 0; i < feats_down_world->size(); i += stride)
        {
            const auto &point = feats_down_world->points[i];
            const std::array<double, 3> endpoint = {point.x, point.y, point.z};
            dynamic_voxel_ray::traverse(
                origin, endpoint, accumulated_map_effective_voxel_size,
                dynamic_map_ray_range, dynamic_map_endpoint_margin,
                [&](const AccumulatedVoxelKey &key) {
                    auto it = accumulated_map_voxels.find(key);
                    if (it == accumulated_map_voxels.end() || it->second.hit_frame == accumulated_frame ||
                        it->second.free_evidence.last_frame == accumulated_frame) return;
                    const auto &old = it->second.point;
                    if (!supported_elevated(old.x, old.y, old.z))
                    {
                        it->second.free_evidence.reset();
                        return;
                    }
                    if (!it->second.free_evidence.observe(
                            accumulated_frame, static_cast<uint8_t>(dynamic_map_clear_frames),
                            static_cast<uint32_t>(dynamic_map_clear_max_gap_frames))) return;
                    erased.push_back(key);
                    accumulated_map_voxels.erase(it);
                });
        }
        const size_t directly_cleared = erased.size();
        std::vector<AccumulatedVoxelKey> tree_erased(erased.begin(), erased.end());
        dynamic_direct_cleared_total += directly_cleared;
        for (size_t i = 0; i < directly_cleared; ++i)
            dynamic_clear_seeds.push_back({erased[i], accumulated_frame});
        while (!dynamic_clear_seeds.empty() &&
               accumulated_frame - dynamic_clear_seeds.front().frame > 15)
            dynamic_clear_seeds.pop_front();
        while (dynamic_clear_seeds.size() > 300) dynamic_clear_seeds.pop_front();

        if (dynamic_map_ground_replace_enabled && dynamic_map_ground_max_points > 0)
        {
            struct CurrentGroundCell { float low; float high; uint16_t returns; };
            std::unordered_map<AccumulatedVoxelKey, CurrentGroundCell, dynamic_voxel_ray::Hash> current_ground;
            current_ground.reserve(feats_down_world->size() / 2 + 1);
            for (const auto &point : feats_down_world->points)
            {
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
                const auto key = accumulated_voxel_key(point, accumulated_map_effective_voxel_size);
                const AccumulatedVoxelKey xy_key{key.x, key.y, 0};
                auto [it, inserted] = current_ground.try_emplace(
                    xy_key, CurrentGroundCell{point.z, point.z, 1});
                if (!inserted)
                {
                    it->second.low = std::min(it->second.low, point.z);
                    it->second.high = std::max(it->second.high, point.z);
                    it->second.returns = static_cast<uint16_t>(std::min<int>(
                        65535, it->second.returns + 1));
                }
            }

            size_t ground_replace_budget = static_cast<size_t>(dynamic_map_ground_max_points);
            const double search_radius_sq = dynamic_map_ground_search_radius * dynamic_map_ground_search_radius;
            const int radius_cells = std::max(1, static_cast<int>(std::ceil(
                dynamic_map_ground_search_radius / accumulated_map_effective_voxel_size)));
            // Only inspect historical voxels above ground cells observed in this
            // scan.  Walking the whole accumulated map here makes the mapping
            // cost grow with session length and eventually breaks real-time use.
            std::unordered_set<AccumulatedVoxelKey, dynamic_voxel_ray::Hash> ground_checked;
            ground_checked.reserve(std::min<size_t>(
                static_cast<size_t>(dynamic_map_ground_max_points) * 4 + 1,
                current_ground.size() * static_cast<size_t>((2 * radius_cells + 1) *
                                                             (2 * radius_cells + 1) * 4) + 1));
            for (const auto &ground_entry : current_ground)
            {
                if (ground_replace_budget == 0) break;
                const auto &ground_key = ground_entry.first;
                const auto &observed = ground_entry.second;
                if (observed.returns < dynamic_map_ground_min_returns ||
                    observed.high - observed.low > dynamic_map_ground_max_span)
                    continue;
                const int64_t z_begin = static_cast<int64_t>(std::floor(
                    (observed.low + dynamic_map_ground_min_height) / accumulated_map_effective_voxel_size));
                const int64_t z_end = static_cast<int64_t>(std::ceil(
                    (observed.low + 3.0) / accumulated_map_effective_voxel_size));
                for (int dy = -radius_cells; dy <= radius_cells && ground_replace_budget > 0; ++dy)
                {
                    for (int dx = -radius_cells; dx <= radius_cells && ground_replace_budget > 0; ++dx)
                    {
                        const auto candidate_x = ground_key.x + dx;
                        const auto candidate_y = ground_key.y + dy;
                        const double gx = (candidate_x + 0.5) * accumulated_map_effective_voxel_size;
                        const double gy = (candidate_y + 0.5) * accumulated_map_effective_voxel_size;
                        const double ground_x = (ground_key.x + 0.5) * accumulated_map_effective_voxel_size;
                        const double ground_y = (ground_key.y + 0.5) * accumulated_map_effective_voxel_size;
                        if ((gx - ground_x) * (gx - ground_x) + (gy - ground_y) * (gy - ground_y) >
                            search_radius_sq)
                            continue;
                        for (int64_t z = z_begin; z <= z_end && ground_replace_budget > 0; ++z)
                        {
                            const AccumulatedVoxelKey candidate{candidate_x, candidate_y, z};
                            if (ground_checked.count(candidate) != 0) continue;
                            const auto it = accumulated_map_voxels.find(candidate);
                            if (it == accumulated_map_voxels.end()) continue;
                            auto &old = it->second;
                            if (old.hit_frame == accumulated_frame || old.point.z < -100.0f) continue;
                            const double old_z = old.point.z;
                            if (old_z - observed.low < dynamic_map_ground_min_height ||
                                old_z - observed.low > 3.0) continue;
                            // Mark only a valid historical candidate.  A nearby
                            // but lower/higher terrain cell must not prevent a
                            // later neighbor from validating the same voxel.
                            ground_checked.insert(candidate);
                            --ground_replace_budget;
                            if (!old.free_evidence.observe(
                                    accumulated_frame, static_cast<uint8_t>(dynamic_map_ground_clear_frames),
                                    static_cast<uint32_t>(dynamic_map_clear_max_gap_frames)))
                                continue;
                            erased.push_back(candidate);
                        }
                    }
                }
            }
            for (const auto &key : erased)
            {
                const auto it = accumulated_map_voxels.find(key);
                if (it != accumulated_map_voxels.end()) accumulated_map_voxels.erase(it);
            }
        }

        if (dynamic_map_expand_enabled && !dynamic_clear_seeds.empty())
        {
            const auto nearby_ground = [&](double x, double y, double &ground_z) {
                const int64_t coarse_x = static_cast<int64_t>(std::floor(x / coarse_size));
                const int64_t coarse_y = static_cast<int64_t>(std::floor(y / coarse_size));
                double nearest_distance_sq = 1.6 * 1.6;
                bool found = false;
                for (int cy = -4; cy <= 4; ++cy)
                {
                    for (int cx = -4; cx <= 4; ++cx)
                    {
                        const auto observed = ground_cells.find({coarse_x + cx, coarse_y + cy, 0});
                        if (observed == ground_cells.end() ||
                            observed->second.high - observed->second.low > 0.20f) continue;
                        const double dx = (coarse_x + cx + 0.5) * coarse_size - x;
                        const double dy = (coarse_y + cy + 0.5) * coarse_size - y;
                        const double distance_sq = dx * dx + dy * dy;
                        if (distance_sq > nearest_distance_sq) continue;
                        nearest_distance_sq = distance_sq;
                        ground_z = observed->second.low;
                        found = true;
                    }
                }
                if (!found) return false;
                int supporting_cells = 0;
                for (int cy = -4; cy <= 4; ++cy)
                {
                    for (int cx = -4; cx <= 4; ++cx)
                    {
                        const auto observed = ground_cells.find({coarse_x + cx, coarse_y + cy, 0});
                        if (observed == ground_cells.end() ||
                            observed->second.high - observed->second.low > 0.20f ||
                            std::abs(observed->second.low - ground_z) > 0.25f) continue;
                        const double dx = (coarse_x + cx + 0.5) * coarse_size - x;
                        const double dy = (coarse_y + cy + 0.5) * coarse_size - y;
                        if (dx * dx + dy * dy <= 1.6 * 1.6) ++supporting_cells;
                    }
                }
                return supporting_cells >= 3;
            };

            // A directly cleared voxel anchors a small historical cluster. Nearby
            // ground need not fall in the exact old obstacle cell after it moves.
            std::vector<DynamicClearSeed> anchors;
            anchors.reserve(32);
            for (auto seed = dynamic_clear_seeds.rbegin(); seed != dynamic_clear_seeds.rend(); ++seed)
            {
                const double sx = (seed->key.x + 0.5) * accumulated_map_effective_voxel_size;
                const double sy = (seed->key.y + 0.5) * accumulated_map_effective_voxel_size;
                int close_seeds = 0;
                uint32_t first_frame = 0;
                bool multiple_frames = false;
                for (const auto &neighbor : dynamic_clear_seeds)
                {
                    if (accumulated_frame - neighbor.frame > 8) continue;
                    const double dx = (neighbor.key.x - seed->key.x) * accumulated_map_effective_voxel_size;
                    const double dy = (neighbor.key.y - seed->key.y) * accumulated_map_effective_voxel_size;
                    const double dz = (neighbor.key.z - seed->key.z) * accumulated_map_effective_voxel_size;
                    if (dx * dx + dy * dy > 1.2 * 1.2 || std::abs(dz) > 1.0) continue;
                    ++close_seeds;
                    if (first_frame == 0) first_frame = neighbor.frame;
                    else if (first_frame != neighbor.frame) multiple_frames = true;
                }
                if (close_seeds < dynamic_map_expand_min_seeds || !multiple_frames) continue;
                bool separated = true;
                for (const auto &anchor : anchors)
                {
                    const double ax = (anchor.key.x + 0.5) * accumulated_map_effective_voxel_size;
                    const double ay = (anchor.key.y + 0.5) * accumulated_map_effective_voxel_size;
                    if (std::hypot(sx - ax, sy - ay) < 0.7) { separated = false; break; }
                }
                if (separated) anchors.push_back(*seed);
                if (anchors.size() == 32) break;
            }
            const int xy_radius = static_cast<int>(std::ceil(2.0 / accumulated_map_effective_voxel_size));
            const int z_radius = static_cast<int>(std::ceil(1.2 / accumulated_map_effective_voxel_size));
            for (const auto &seed : anchors)
            {
                const double seed_x = (seed.key.x + 0.5) * accumulated_map_effective_voxel_size;
                const double seed_y = (seed.key.y + 0.5) * accumulated_map_effective_voxel_size;
                const double seed_z = (seed.key.z + 0.5) * accumulated_map_effective_voxel_size;
                std::unordered_set<AccumulatedVoxelKey, dynamic_voxel_ray::Hash> candidates;
                std::unordered_set<AccumulatedVoxelKey, dynamic_voxel_ray::Hash> cleared_neighbors;
                for (const auto &near_seed : dynamic_clear_seeds)
                {
                    const double dx = (near_seed.key.x - seed.key.x) * accumulated_map_effective_voxel_size;
                    const double dy = (near_seed.key.y - seed.key.y) * accumulated_map_effective_voxel_size;
                    const double dz = (near_seed.key.z - seed.key.z) * accumulated_map_effective_voxel_size;
                    if (dx * dx + dy * dy <= 4.0 && std::abs(dz) <= 1.2)
                        cleared_neighbors.insert(near_seed.key);
                }
                for (int dx = -xy_radius; dx <= xy_radius; ++dx)
                {
                    for (int dy = -xy_radius; dy <= xy_radius; ++dy)
                    {
                        for (int dz = -z_radius; dz <= z_radius; ++dz)
                        {
                            const AccumulatedVoxelKey key{seed.key.x + dx, seed.key.y + dy, seed.key.z + dz};
                            auto it = accumulated_map_voxels.find(key);
                            if (it == accumulated_map_voxels.end() ||
                                it->second.hit_frame + 2 > accumulated_frame ||
                                it->second.ground_free_frame == accumulated_frame) continue;
                            const auto &old = it->second.point;
                            if (std::hypot(old.x - seed_x, old.y - seed_y) > 2.0 ||
                                std::abs(old.z - seed_z) > 1.2) continue;
                            double ground_z = 0.0;
                            if (!nearby_ground(old.x, old.y, ground_z) || old.z - ground_z < 0.25 ||
                                old.z - ground_z > 2.0) continue;
                            candidates.insert(key);
                        }
                    }
                }
                dynamic_voxel_ray::flood_connected(
                    seed.key,
                    [&](const AccumulatedVoxelKey &key) {
                        return candidates.count(key) != 0 || cleared_neighbors.count(key) != 0;
                    },
                    [&](const AccumulatedVoxelKey &key) {
                        if (candidates.count(key) == 0) return;
                        auto it = accumulated_map_voxels.find(key);
                        if (it == accumulated_map_voxels.end()) return;
                        if (it->second.ground_free_frame + 1 != accumulated_frame)
                            it->second.ground_free_scans = 0;
                        it->second.ground_free_frame = accumulated_frame;
                        // The seed itself already carries multi-frame direct
                        // free-space evidence; one subsequent ground-supported
                        // scan is sufficient for the published-history copy.
                        if (++it->second.ground_free_scans >= 1)
                        {
                            erased.push_back(key);
                            accumulated_map_voxels.erase(it);
                        }
                    },
                    6000);
            }
        }

        dynamic_ground_cleared_total += erased.size() - directly_cleared;
        // Ground-proximity propagation only edits the published history.
        // The registration tree needs direct free-ray evidence before deletion.
        for (const auto &key : tree_erased)
        {
            BoxPointType box;
            for (int axis = 0; axis < 3; ++axis)
            {
                const int64_t coord = axis == 0 ? key.x : (axis == 1 ? key.y : key.z);
                box.vertex_min[axis] = static_cast<float>(coord * accumulated_map_effective_voxel_size);
                box.vertex_max[axis] = static_cast<float>((coord + 1) * accumulated_map_effective_voxel_size);
            }
            pending_dynamic_deletions.push_back(box);
        }
        if (!pending_dynamic_deletions.empty() && ikdtree.Root_Node != nullptr)
        {
            std::vector<BoxPointType> boxes;
            const size_t count = std::min<size_t>(256, pending_dynamic_deletions.size());
            boxes.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                boxes.push_back(pending_dynamic_deletions.front());
                pending_dynamic_deletions.pop_front();
            }
            ikdtree.Delete_Point_Boxes(boxes);
        }
    }
    enforce_accumulated_map_budget();
}

void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = map_frame;
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = body_frame;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = map_frame;
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    if (accumulated_map_voxels.empty()) return;
    refresh_accumulated_map_cloud();

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = map_frame;
    pubLaserCloudMap->publish(laserCloudmsg);

    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

void publish_local_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudLocalMap)
{
    if (!local_map_pub_en || ikdtree.Root_Node == nullptr)
        return;

    BoxPointType local_box;
    const float cx = static_cast<float>(state_point.pos(0));
    const float cy = static_cast<float>(state_point.pos(1));
    const float cz = static_cast<float>(state_point.pos(2));
    const float radius = static_cast<float>(std::max(0.5, local_map_radius));
    const float z_half = static_cast<float>(std::max(0.5, local_map_z_half));

    local_box.vertex_min[0] = cx - radius;
    local_box.vertex_max[0] = cx + radius;
    local_box.vertex_min[1] = cy - radius;
    local_box.vertex_max[1] = cy + radius;
    local_box.vertex_min[2] = cz - z_half;
    local_box.vertex_max[2] = cz + z_half;

    PointVector local_points;
    ikdtree.Box_Search(local_box, local_points);
    if (local_points.empty())
        return;

    PointCloudXYZI local_cloud;
    local_cloud.points = std::move(local_points);
    local_cloud.width = local_cloud.points.size();
    local_cloud.height = 1;
    local_cloud.is_dense = false;

    sensor_msgs::msg::PointCloud2 local_cloud_msg;
    pcl::toROSMsg(local_cloud, local_cloud_msg);
    local_cloud_msg.header.stamp = get_ros_time(lidar_end_time);
    local_cloud_msg.header.frame_id = map_frame;
    pubLaserCloudLocalMap->publish(local_cloud_msg);
}

void save_to_pcd()
{
    refresh_accumulated_map_cloud();
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    const M3D rot_pub = state_point.rot.toRotationMatrix() * Publish_R_wrt_IMU;
    const V3D pos_pub = state_point.pos + state_point.rot * Publish_T_wrt_IMU;
    const Eigen::Quaterniond quat_pub(rot_pub);

    out.pose.position.x = pos_pub(0);
    out.pose.position.y = pos_pub(1);
    out.pose.position.z = pos_pub(2);
    out.pose.orientation.x = quat_pub.x();
    out.pose.orientation.y = quat_pub.y();
    out.pose.orientation.z = quat_pub.z();
    out.pose.orientation.w = quat_pub.w();
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = map_frame;
    odomAftMapped.child_frame_id = body_frame;
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = map_frame;
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = body_frame;
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = map_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<bool>("publish.local_map_en", true);
        this->declare_parameter<double>("publish.local_map_radius", 12.0);
        this->declare_parameter<double>("publish.local_map_z_half", 3.0);
        this->declare_parameter<double>("publish.map_publish_hz", 1.0);
        this->declare_parameter<double>("publish.map_voxel_size", 0.30);
        this->declare_parameter<int>("publish.map_max_points", 800000);
        this->declare_parameter<bool>("publish.map_dense_en", false);
        this->declare_parameter<bool>("mapping.dynamic_clear_enabled", false);
        this->declare_parameter<bool>("mapping.dynamic_expand_enabled", false);
        this->declare_parameter<int>("mapping.dynamic_clear_frames", 3);
        this->declare_parameter<int>("mapping.dynamic_clear_max_gap_frames", 3);
        this->declare_parameter<bool>("mapping.dynamic_ground_replace_enabled", false);
        this->declare_parameter<int>("mapping.dynamic_ground_clear_frames", 3);
        this->declare_parameter<int>("mapping.dynamic_ground_min_returns", 2);
        this->declare_parameter<double>("mapping.dynamic_ground_max_span", 0.14);
        this->declare_parameter<double>("mapping.dynamic_ground_search_radius", 0.24);
        this->declare_parameter<double>("mapping.dynamic_ground_min_height", 0.30);
        this->declare_parameter<int>("mapping.dynamic_ground_max_points", 6000);
        this->declare_parameter<int>("mapping.dynamic_expand_min_seeds", 6);
        this->declare_parameter<int>("mapping.dynamic_max_rays", 2500);
        this->declare_parameter<double>("mapping.dynamic_ray_range", 30.0);
        this->declare_parameter<double>("mapping.dynamic_endpoint_margin", 0.65);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<bool>("preprocess.synthetic_point_time", true);
        this->declare_parameter<bool>("preprocess.self_filter_enable", false);
        this->declare_parameter<double>("preprocess.self_filter_min_x", -0.8);
        this->declare_parameter<double>("preprocess.self_filter_max_x", 0.8);
        this->declare_parameter<double>("preprocess.self_filter_min_y", -0.45);
        this->declare_parameter<double>("preprocess.self_filter_max_y", 0.45);
        this->declare_parameter<double>("preprocess.self_filter_min_z", -0.8);
        this->declare_parameter<double>("preprocess.self_filter_max_z", 0.25);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<string>("frame.map_frame", "camera_init");
        this->declare_parameter<string>("frame.body_frame", "body");
        this->declare_parameter<vector<double>>("frame.body_pose_T", vector<double>());
        this->declare_parameter<vector<double>>("frame.body_pose_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<bool>("publish.local_map_en", local_map_pub_en, true);
        this->get_parameter_or<double>("publish.local_map_radius", local_map_radius, 12.0);
        this->get_parameter_or<double>("publish.local_map_z_half", local_map_z_half, 3.0);
        this->get_parameter_or<double>("publish.map_publish_hz", accumulated_map_publish_hz, 1.0);
        this->get_parameter_or<double>("publish.map_voxel_size", accumulated_map_voxel_size, 0.30);
        this->get_parameter_or<int>("publish.map_max_points", accumulated_map_max_points, 800000);
        this->get_parameter_or<bool>("publish.map_dense_en", accumulated_map_dense, false);
        this->get_parameter_or<bool>("mapping.dynamic_clear_enabled", dynamic_map_clear_enabled, false);
        this->get_parameter_or<bool>("mapping.dynamic_expand_enabled", dynamic_map_expand_enabled, false);
        this->get_parameter_or<int>("mapping.dynamic_clear_frames", dynamic_map_clear_frames, 3);
        this->get_parameter_or<int>("mapping.dynamic_clear_max_gap_frames", dynamic_map_clear_max_gap_frames, 3);
        this->get_parameter_or<bool>("mapping.dynamic_ground_replace_enabled", dynamic_map_ground_replace_enabled, false);
        this->get_parameter_or<int>("mapping.dynamic_ground_clear_frames", dynamic_map_ground_clear_frames, 3);
        this->get_parameter_or<int>("mapping.dynamic_ground_min_returns", dynamic_map_ground_min_returns, 2);
        this->get_parameter_or<double>("mapping.dynamic_ground_max_span", dynamic_map_ground_max_span, 0.14);
        this->get_parameter_or<double>("mapping.dynamic_ground_search_radius", dynamic_map_ground_search_radius, 0.24);
        this->get_parameter_or<double>("mapping.dynamic_ground_min_height", dynamic_map_ground_min_height, 0.30);
        this->get_parameter_or<int>("mapping.dynamic_ground_max_points", dynamic_map_ground_max_points, 6000);
        this->get_parameter_or<int>("mapping.dynamic_expand_min_seeds", dynamic_map_expand_min_seeds, 6);
        this->get_parameter_or<int>("mapping.dynamic_max_rays", dynamic_map_max_rays, 2500);
        this->get_parameter_or<double>("mapping.dynamic_ray_range", dynamic_map_ray_range, 30.0);
        this->get_parameter_or<double>("mapping.dynamic_endpoint_margin", dynamic_map_endpoint_margin, 0.65);
        dynamic_map_clear_frames = std::clamp(dynamic_map_clear_frames, 2, 10);
        dynamic_map_clear_max_gap_frames = std::clamp(dynamic_map_clear_max_gap_frames, 1, 10);
        dynamic_map_ground_clear_frames = std::clamp(dynamic_map_ground_clear_frames, 2, 10);
        dynamic_map_ground_min_returns = std::clamp(dynamic_map_ground_min_returns, 1, 20);
        dynamic_map_ground_max_span = std::clamp(dynamic_map_ground_max_span, 0.03, 0.5);
        dynamic_map_ground_search_radius = std::clamp(dynamic_map_ground_search_radius, 0.05, 0.6);
        dynamic_map_ground_min_height = std::clamp(dynamic_map_ground_min_height, 0.05, 1.0);
        dynamic_map_ground_max_points = std::clamp(dynamic_map_ground_max_points, 0, 50000);
        dynamic_map_expand_min_seeds = std::clamp(dynamic_map_expand_min_seeds, 2, 32);
        dynamic_map_max_rays = std::clamp(dynamic_map_max_rays, 0, 10000);
        dynamic_map_ray_range = std::clamp(dynamic_map_ray_range, 2.0, 100.0);
        dynamic_map_endpoint_margin = std::clamp(dynamic_map_endpoint_margin, 0.3, 2.0);
        RCLCPP_INFO(this->get_logger(),
                    "Dynamic map policy: direct=%s frames=%d max_gap=%d ground=%s ground_frames=%d min_height=%.2f expansion=%s min_seeds=%d",
                    dynamic_map_clear_enabled ? "on" : "off", dynamic_map_clear_frames,
                    dynamic_map_clear_max_gap_frames, dynamic_map_ground_replace_enabled ? "on" : "off",
                    dynamic_map_ground_clear_frames, dynamic_map_ground_min_height,
                    dynamic_map_expand_enabled ? "on" : "off", dynamic_map_expand_min_seeds);
        accumulated_map_effective_voxel_size = std::max(0.05, accumulated_map_voxel_size);
        accumulated_map_voxels.reserve(static_cast<size_t>(std::min(200000, std::max(0, accumulated_map_max_points))));
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<bool>("preprocess.synthetic_point_time", p_pre->synthetic_point_time_enable, true);
        this->get_parameter_or<bool>("preprocess.self_filter_enable", p_pre->self_filter_enable, false);
        this->get_parameter_or<double>("preprocess.self_filter_min_x", p_pre->self_filter_min_x, -0.8);
        this->get_parameter_or<double>("preprocess.self_filter_max_x", p_pre->self_filter_max_x, 0.8);
        this->get_parameter_or<double>("preprocess.self_filter_min_y", p_pre->self_filter_min_y, -0.45);
        this->get_parameter_or<double>("preprocess.self_filter_max_y", p_pre->self_filter_max_y, 0.45);
        this->get_parameter_or<double>("preprocess.self_filter_min_z", p_pre->self_filter_min_z, -0.8);
        this->get_parameter_or<double>("preprocess.self_filter_max_z", p_pre->self_filter_max_z, 0.25);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<string>("frame.map_frame", map_frame, "camera_init");
        this->get_parameter_or<string>("frame.body_frame", body_frame, "body");
        this->get_parameter_or<vector<double>>("frame.body_pose_T", publish_body_T_param, vector<double>());
        this->get_parameter_or<vector<double>>("frame.body_pose_R", publish_body_R_param, vector<double>());

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);
        RCLCPP_INFO(this->get_logger(),
                    "p_pre synthetic PointCloud2 point time: %s",
                    p_pre->synthetic_point_time_enable ? "true" : "false");
        if (p_pre->self_filter_enable)
        {
            RCLCPP_INFO(this->get_logger(),
                        "p_pre self filter box x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]",
                        p_pre->self_filter_min_x, p_pre->self_filter_max_x,
                        p_pre->self_filter_min_y, p_pre->self_filter_max_y,
                        p_pre->self_filter_min_z, p_pre->self_filter_max_z);
        }

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = map_frame;

        // /*** variables definition ***/
        // int effect_feat_num = 0, frame_num = 0;
        // double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        // bool flg_EKF_converged, EKF_stop_flg = 0;

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        if (publish_body_T_param.size() == 3)
        {
            Publish_T_wrt_IMU << VEC_FROM_ARRAY(publish_body_T_param);
        }
        else
        {
            Publish_T_wrt_IMU.setZero();
            RCLCPP_WARN(this->get_logger(), "frame.body_pose_T must contain 3 values. Falling back to identity pose offset.");
        }
        if (publish_body_R_param.size() == 9)
        {
            Publish_R_wrt_IMU << MAT_FROM_ARRAY(publish_body_R_param);
        }
        else
        {
            Publish_R_wrt_IMU.setIdentity();
            RCLCPP_WARN(this->get_logger(), "frame.body_pose_R must contain 9 values. Falling back to identity rotation offset.");
        }
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        // FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        // ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        // Keep the latest global map available to RViz and to analysis nodes
        // that are started after FAST-LIO has already initialized.
        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1));
        map_qos.reliable().transient_local();
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", map_qos);
        pubLaserCloudLocalMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_local_map", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        //------------------------------------------------------------------------------------------------------
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        const double safe_map_publish_hz = std::max(0.05, accumulated_map_publish_hz);
        if (map_pub_en)
        {
            const auto map_period_ms = std::chrono::milliseconds(
                std::max<int64_t>(1, static_cast<int64_t>(1000.0 / safe_map_publish_hz)));
            // Map visualization must not depend on a simulated sensor stamp
            // advancing. The scan callback only updates the accumulated data;
            // this wall timer republishes the latest snapshot independently.
            map_pub_timer_ = this->create_wall_timer(
                map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));
        }
        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        if (map_pub_en)
        {
            RCLCPP_INFO(this->get_logger(),
                        "Accumulated map enabled: topic=/Laser_map rate=%.2fHz voxel=%.2fm max_points=%d source=%s",
                        safe_map_publish_hz, accumulated_map_effective_voxel_size,
                        std::max(0, accumulated_map_max_points),
                        accumulated_map_dense ? "deskewed_full_scan" : "registration_scan");
        }

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

private:
    void timer_callback()
    {
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                    if (map_pub_en || pcd_save_en || dynamic_map_clear_enabled) accumulate_map_frame();
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            if (map_pub_en || pcd_save_en || dynamic_map_clear_enabled) accumulate_map_frame();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            if (local_map_pub_en) publish_local_map(pubLaserCloudLocalMap_);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Cloud detail: deskewed=%zu registration=%d accumulated=%zu voxel=%.3fm processing=%.1fms",
                feats_undistort->size(), feats_down_size, accumulated_map_voxels.size(),
                accumulated_map_effective_voxel_size, 1000.0 * (omp_get_wtime() - t0));
            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }
    }

    void map_publish_callback()
    {
        if (!map_pub_en || accumulated_map_voxels.empty()) return;
        publish_map(pubLaserCloudMap_);
        if (dynamic_map_clear_enabled &&
            (dynamic_direct_cleared_total != last_logged_direct_clear_ ||
             dynamic_ground_cleared_total != last_logged_ground_clear_))
        {
            RCLCPP_INFO(this->get_logger(),
                        "Dynamic map clearing: direct=%lu expanded=%lu pending_ikd=%zu",
                        static_cast<unsigned long>(dynamic_direct_cleared_total),
                        static_cast<unsigned long>(dynamic_ground_cleared_total),
                        pending_dynamic_deletions.size());
            last_logged_direct_clear_ = dynamic_direct_cleared_total;
            last_logged_ground_clear_ = dynamic_ground_cleared_total;
        }
        if (!map_publish_logged_)
        {
            RCLCPP_INFO(this->get_logger(), "First accumulated map published: %zu points", accumulated_map_voxels.size());
            map_publish_logged_ = true;
        }
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (pcd_save_en)
        {
            save_to_pcd();
            res->success = true;
            res->message = "Map saved.";
        }
        else
        {
            res->success = false;
            res->message = "Map save disabled.";
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudLocalMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en = false, map_pub_en = false;
    bool map_publish_logged_ = false;
    uint64_t last_logged_direct_clear_ = 0;
    uint64_t last_logged_ground_clear_ = 0;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
