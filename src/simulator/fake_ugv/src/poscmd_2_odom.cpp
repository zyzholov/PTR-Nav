#include <iostream>
#include <math.h>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/empty.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>

using namespace std;

// Global node pointer
std::shared_ptr<rclcpp::Node> g_node;

rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub;
rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr traj_sub;
rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;
rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr  odom_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_pub;
rclcpp::TimerBase::SharedPtr cloud_timer;

geometry_msgs::msg::Twist cmd_vel_msg;
sensor_msgs::msg::PointCloud2 world_cloud_msg;
nav_msgs::msg::Odometry new_odom;
visualization_msgs::msg::Marker diff_marker;
rclcpp::Time begin_time;

pcl::PointCloud<pcl::PointXYZ>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXY>::Ptr world_cloud_plane(new pcl::PointCloud<pcl::PointXY>());
pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
pcl::KdTreeFLANN<pcl::PointXY> kd_tree_plane;
std::string map_name;
Eigen::Vector3d now_p;
Eigen::Matrix3d now_R;
double p_init_x, p_init_y, p_init_z, yaw_init, max_height;
double time_resolution = 0.01;
bool rcv_cmd = false;
double x = 0.0;
double y = 0.0;
double v = 0.0;
double steer = 0.0;
double yaw = 0.0;
double vx = 0.0;
double vy = 0.0;
double w = 0.0;
double filtered_height_z = 0.0;
bool has_filtered_height = false;
double z_deadband = 0.01;
double z_max_rate = 0.5;
bool require_initialpose = true;
bool pose_initialized = false;
bool marker_hidden_sent = false;
bool allow_runtime_reinit = true;
double reinit_min_interval = 0.5;
double reinit_max_move_when_running = 2.0;
double max_position_step = 0.08;
double max_yaw_step = 0.25;
double max_cmd_speed = 1.5;
double min_cmd_speed = -0.8;
double max_cmd_accel = 1.0;
double max_cmd_steer = 0.6;
double max_cmd_steer_rate = 1.0;
rclcpp::Time last_initialpose_time;
bool has_initialpose_time = false;
bool goal_active = false;
bool goal_hold = false;
bool enable_goal_watchdog = true;
double goal_x = 0.0;
double goal_y = 0.0;
double goal_stop_radius = 0.35;
double goal_release_radius = 0.60;
double goal_abort_radius = 25.0;
double goal_no_progress_sec = 3.0;
double goal_progress_eps = 0.03;
double last_goal_dist = 0.0;
bool has_last_goal_dist = false;
rclcpp::Time last_goal_progress_time;
bool has_goal_progress_time = false;
Eigen::Quaterniond filtered_att_q = Eigen::Quaterniond::Identity();
bool has_filtered_att = false;
double att_filter_alpha = 0.18;
double map_min_x = -1e9;
double map_max_x = 1e9;
double map_min_y = -1e9;
double map_max_y = 1e9;
bool has_map_bounds = false;
double map_bound_margin = 0.3;
rclcpp::Time last_cmd_time;
bool has_last_cmd_time = false;
double cmd_timeout_sec = 0.4;
double last_ground_x = 0.0;
double last_ground_y = 0.0;
double last_ground_z = 0.0;
bool has_last_ground_z = false;

// simulator parameters
// double init_x = 0.0;
// double init_y = 0.0;
// double init_v = 0.0;
// double init_steer = 0.0;
// double init_yaw = 0.0;
double max_steer = 0.7;
double max_steer_vel = 7.0;
double time_delay = 0.0;
double wheel_base = 0.6;
// double noise_percent = 0.0;
// double noise_percent = 0.1;
Eigen::Quaterniond q_mesh;
Eigen::Vector3d pos_mesh;

double terrain_to_body_height(double terrain_z)
{
	// Keep body at a fixed offset above terrain to avoid z oscillation from normal-vector changes.
	return terrain_z + max_height;
}

double robust_ground_height_from_indices(const std::vector<int> &idxs, double fallback_z, double query_x, double query_y)
{
	if (idxs.empty())
	{
		return fallback_z;
	}

	std::vector<double> z_values;
	z_values.reserve(idxs.size());
	for (int id : idxs)
	{
		const double z = world_cloud->points[id].z;
		if (std::isfinite(z))
		{
			z_values.push_back(z);
		}
	}
	if (z_values.empty())
	{
		return fallback_z;
	}

	std::sort(z_values.begin(), z_values.end());
	const size_t n = z_values.size();
	auto q = [&](double r) -> double {
		const size_t qi = std::min(n - 1, static_cast<size_t>(std::floor(r * static_cast<double>(n - 1))));
		return z_values[qi];
	};

	const double q05 = q(0.05);
	const double q20 = q(0.20);
	const double q35 = q(0.35);

	// Average low-band heights to stay on ground and suppress canopy points.
	const double low_lo = q05 - 0.05;
	const double low_hi = q35 + 0.08;
	double sum = 0.0;
	int cnt = 0;
	for (double z : z_values)
	{
		if (z >= low_lo && z <= low_hi)
		{
			sum += z;
			cnt++;
		}
	}
	double ground_z = (cnt >= 3) ? (sum / static_cast<double>(cnt)) : q20;

	if (has_last_ground_z)
	{
		const double dxy = std::hypot(query_x - last_ground_x, query_y - last_ground_y);
		const double rise_limit = 0.08 + 0.45 * dxy;
		const double drop_limit = 0.12 + 0.70 * dxy;
		const double lo = last_ground_z - std::max(0.12, drop_limit);
		const double hi = last_ground_z + std::max(0.10, rise_limit);
		ground_z = std::clamp(ground_z, lo, hi);
	}

	last_ground_x = query_x;
	last_ground_y = query_y;
	last_ground_z = ground_z;
	has_last_ground_z = true;
	return ground_z;
}


Eigen::Vector4d getTPM(Eigen::Vector3d pos, vector<Eigen::Vector3d> points)
{
	Eigen::Vector4d tpm;

	Eigen::Vector3d mean_points = Eigen::Vector3d::Zero();
	for (size_t i=0; i<points.size(); i++)
		mean_points+=points[i];

	mean_points /= (double)points.size();

	Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
	for (size_t i=0; i<points.size(); i++)
	{
		Eigen::Vector3d v = points[i] - mean_points;
		cov += v * v.transpose();
	}
	cov /= (double)points.size();
	Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
	Eigen::Matrix<double, 3, 1> D = es.pseudoEigenvalueMatrix().diagonal();// eigenValue
	Eigen::Matrix3d V = es.pseudoEigenvectors();    // eigenVector
	Eigen::MatrixXd::Index evalsMax;
	D.minCoeff(&evalsMax);
	Eigen::Matrix<double, 3, 1> n = V.col(evalsMax);
	n.normalize();
	if (n(2, 0) < 0.0)
		n = -n;

	tpm[3] = D(evalsMax) / D.sum() * 3.0;
	if (isnan(tpm[3]))
		n = Eigen::Vector3d(1.0, 0.0, 0.0);
	tpm[3] = mean_points.z();
	tpm[0] = n(0, 0);
	tpm[1] = n(1, 0);
	tpm[2] = n(2, 0);

	return tpm;
}

void getSE3(const Eigen::Vector3d& se2_odom, 
			Eigen::Vector3d& p,
			Eigen::Matrix3d& R)
{
	p(0) = se2_odom(0);
	p(1) = se2_odom(1);
	
	// If world cloud is empty, just use flat ground assumption
	if (world_cloud->points.empty())
	{
		// Simple SE3: identity rotation, flat ground
		p(2) = 0.0;
		Eigen::Quaterniond q(cos(se2_odom(2)), 0.0, 0.0, sin(se2_odom(2)));
		R = q.toRotationMatrix();
		return;
	}
	
	vector<int> Idxs;
	vector<float> SquaredDists;
	pcl::PointXY pxy;
	pxy.x = se2_odom(0);
	pxy.y = se2_odom(1);
	vector<Eigen::Vector3d> points;
	if (kd_tree_plane.radiusSearch(pxy, 0.8, Idxs, SquaredDists) > 0)
	{
		const double ground_z = robust_ground_height_from_indices(Idxs, p(2), se2_odom(0), se2_odom(1));
		p(2) = ground_z;

		for (size_t i = 0; i < Idxs.size(); ++i)
		{
			const auto &wp = world_cloud->points[Idxs[i]];
			if (wp.z <= ground_z + 0.22 && wp.z >= ground_z - 0.20)
				points.emplace_back(Eigen::Vector3d(wp.x, wp.y, wp.z));
		}
	}
	else if (kd_tree_plane.nearestKSearch(pxy, 24, Idxs, SquaredDists) > 0)
	{
		const double ground_z = robust_ground_height_from_indices(Idxs, p(2), se2_odom(0), se2_odom(1));
		p(2) = ground_z;
		for (size_t i = 0; i < Idxs.size(); ++i)
		{
			const auto &wp = world_cloud->points[Idxs[i]];
			if (wp.z <= ground_z + 0.20 && wp.z >= ground_z - 0.20)
			{
				points.emplace_back(Eigen::Vector3d(wp.x, wp.y, wp.z));
			}
		}
	}
	else
	{
		RCLCPP_WARN(g_node->get_logger(), "no points in the map");
	}
	if (points.empty())
	{
		pcl::PointXYZ pt;
		pt.x = se2_odom(0);
		pt.y = se2_odom(1);
		pt.z = p(2);
		if (kd_tree.radiusSearch(pt, 0.5, Idxs, SquaredDists) > 0)
		{
			for (size_t i=0; i<Idxs.size(); i++)
				points.emplace_back(Eigen::Vector3d(world_cloud->points[Idxs[i]].x, \
											world_cloud->points[Idxs[i]].y, \
											world_cloud->points[Idxs[i]].z ));
		}
	}
	if (points.empty())
	{	
		Eigen::Quaterniond q(cos(se2_odom(2)), 0.0, 0.0, sin(se2_odom(2)));
		R = q.toRotationMatrix();
	}
	else
	{
		Eigen::Vector4d tpm = getTPM(p, points);
		Eigen::Vector3d xyaw(cos(se2_odom(2)), sin(se2_odom(2)), 0.0);
		Eigen::Vector3d zb = tpm.head(3);
		Eigen::Vector3d yb = zb.cross(xyaw).normalized();
		Eigen::Vector3d xb = yb.cross(zb);
		R.col(0) = xb;
		R.col(1) = yb;
		R.col(2) = zb;
		// Keep height from nearest-point lookup and use plane fit only for orientation.
	}
	return;
}

void initParams()
{
	// global map
	pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);   // hxlin fixed : change global map
	pcl::PLYReader ply_reader;
	pcl::PCDReader pcd_reader;
	std::vector<std::string> map_candidates;
	if (!map_name.empty())
	{
		map_candidates.push_back(map_name);
	}
	try
	{
		const std::string fake_ugv_share = ament_index_cpp::get_package_share_directory("fake_ugv");
		map_candidates.push_back(fake_ugv_share + "/maps/" + map_name);
	}
	catch (const std::exception &)
	{
		// Keep direct-path candidate only when package index is unavailable.
	}

	std::string map_path;
	for (const auto &candidate : map_candidates)
	{
		if (!candidate.empty() && std::filesystem::exists(candidate))
		{
			map_path = candidate;
			break;
		}
	}
	if (map_path.empty() && !map_candidates.empty())
	{
		map_path = map_candidates.front();
	}

	int load_result = pcl::io::loadPCDFile<pcl::PointXYZ>(map_path, *world_cloud); // load global map
	if (load_result == -1)
	{
		   printf("\nCouldn't read map file: %s (map_name=%s). Running without terrain map.\n\n",
				  map_path.c_str(), map_name.c_str());
		// Continue without map - just simulate odometry
		world_cloud->clear();
	}
	else
	{
		RCLCPP_INFO(g_node->get_logger(), "Loaded terrain map: %s (%zu points)", map_path.c_str(), world_cloud->size());
	}
	world_cloud->width = world_cloud->points.size();
	world_cloud->height = 1;
	world_cloud->is_dense = true;
	world_cloud->header.frame_id = "map";
	kd_tree.setInputCloud(world_cloud);
	pcl::toROSMsg(*world_cloud, world_cloud_msg);

	if (!world_cloud->points.empty())
	{
		map_min_x = map_max_x = world_cloud->points.front().x;
		map_min_y = map_max_y = world_cloud->points.front().y;
		for (const auto &pt : world_cloud->points)
		{
			map_min_x = std::min(map_min_x, (double)pt.x);
			map_max_x = std::max(map_max_x, (double)pt.x);
			map_min_y = std::min(map_min_y, (double)pt.y);
			map_max_y = std::max(map_max_y, (double)pt.y);
		}
		has_map_bounds = true;
	}

	for (size_t i=0; i<world_cloud->points.size(); i++)
	{
		pcl::PointXY p;
		p.x = world_cloud->points[i].x;
		p.y = world_cloud->points[i].y;
		world_cloud_plane->points.push_back(p);
	}
	world_cloud_plane->width = world_cloud_plane->points.size();
	world_cloud_plane->height = 1;
	world_cloud_plane->is_dense = true;
	world_cloud_plane->header.frame_id = "map";
	kd_tree_plane.setInputCloud(world_cloud_plane);

	// find initial position
	now_p.z() = p_init_z;
	getSE3(Eigen::Vector3d(p_init_x, p_init_y, yaw_init), now_p, now_R);
	new_odom.header.stamp    = g_node->get_clock()->now();
	new_odom.header.frame_id = "map";
	filtered_height_z = terrain_to_body_height(now_p.z());
	has_filtered_height = true;
	new_odom.pose.pose.position.x = x = now_p.x();
	new_odom.pose.pose.position.y = y = now_p.y();
	new_odom.pose.pose.position.z = filtered_height_z;

	yaw = yaw_init;
	Eigen::Quaterniond q(now_R);
	new_odom.pose.pose.orientation.w = q.w();
	new_odom.pose.pose.orientation.x = q.x();
	new_odom.pose.pose.orientation.y = q.y();
	new_odom.pose.pose.orientation.z = q.z();
	new_odom.twist.twist.linear.x = 0.0;
	new_odom.twist.twist.linear.y = 0.0;
	new_odom.twist.twist.linear.z = 0.0;
	new_odom.twist.twist.angular.x = 0.0;
	new_odom.twist.twist.angular.y = 0.0;
	new_odom.twist.twist.angular.z = 0.0;

	diff_marker.header.frame_id = "map";
	diff_marker.id = 0;
	diff_marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
	diff_marker.action = visualization_msgs::msg::Marker::ADD;
	diff_marker.pose = new_odom.pose.pose;
	diff_marker.color.a = 1.0;
	diff_marker.color.r = 0.5;
	diff_marker.color.g = 0.5;
	diff_marker.color.b = 0.5;
	diff_marker.scale.x = 0.35;
	diff_marker.scale.y = 0.35;
	diff_marker.scale.z = 0.35;
	diff_marker.mesh_resource = "package://fake_ugv/meshes/car.dae";
}

void rcvVelCmdCallBack(const geometry_msgs::msg::Twist::SharedPtr cmd)
{	
	rcv_cmd 	= true;
	cmd_vel_msg = *cmd;
	last_cmd_time = g_node->get_clock()->now();
	has_last_cmd_time = true;
}

void rcvGoalCallBack(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	if (!enable_goal_watchdog)
	{
		return;
	}

	goal_x = msg->pose.position.x;
	goal_y = msg->pose.position.y;
	goal_active = true;
	goal_hold = false;
	has_last_goal_dist = false;
	has_goal_progress_time = false;
	RCLCPP_INFO(g_node->get_logger(), "Goal updated: x=%.2f y=%.2f", goal_x, goal_y);
}

void rcvInitialPoseCallBack(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg) {
	auto now_t = g_node->get_clock()->now();
	if (pose_initialized)
	{
		if (!allow_runtime_reinit)
		{
			RCLCPP_WARN(g_node->get_logger(), "Ignoring runtime /initialpose (allow_runtime_reinit=false)");
			return;
		}
		if (has_initialpose_time && (now_t - last_initialpose_time).seconds() < reinit_min_interval)
		{
			RCLCPP_WARN(g_node->get_logger(), "Ignoring /initialpose due to debounce window");
			return;
		}
		double req_x = pose_msg->pose.pose.position.x;
		double req_y = pose_msg->pose.pose.position.y;
		double jump_dist = std::hypot(req_x - x, req_y - y);
		if (std::abs(v) > 0.15 && jump_dist > reinit_max_move_when_running)
		{
			RCLCPP_WARN(g_node->get_logger(),
				"Ignoring /initialpose jump while moving (v=%.2f, jump=%.2f m)", v, jump_dist);
			return;
		}
	}

	x = pose_msg->pose.pose.position.x;
	y = pose_msg->pose.pose.position.y;
	double qw = pose_msg->pose.pose.orientation.w;
	double qz = pose_msg->pose.pose.orientation.z;
	yaw = 2.0 * atan2(qz, qw);
	v = 0.0; steer = 0.0; vx = 0.0; vy = 0.0; w = 0.0;
	pose_initialized = true;
	last_initialpose_time = now_t;
	has_initialpose_time = true;
	marker_hidden_sent = false;
	goal_hold = false;
	rcv_cmd = false;
	cmd_vel_msg.linear.x = 0.0;
	cmd_vel_msg.angular.z = 0.0;
	last_cmd_time = now_t;
	has_last_cmd_time = true;
	RCLCPP_INFO(g_node->get_logger(), "Initial pose set: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw);
	
	// Update odometry and marker immediately
	new_odom.header.stamp = g_node->get_clock()->now();
	new_odom.header.frame_id = "map";
	new_odom.pose.pose.position.x  = x;
	new_odom.pose.pose.position.y  = y;
	getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
	// Initial pose reset should snap deterministically, not blend with previous run state.
	filtered_height_z = terrain_to_body_height(now_p.z());
	has_filtered_height = true;
	new_odom.pose.pose.position.z  = filtered_height_z;
	Eigen::Quaterniond q_body(now_R);
	q_body.normalize();
	filtered_att_q = q_body;
	has_filtered_att = true;
	new_odom.pose.pose.orientation.w  = filtered_att_q.w();
	new_odom.pose.pose.orientation.x  = filtered_att_q.x();
	new_odom.pose.pose.orientation.y  = filtered_att_q.y();
	new_odom.pose.pose.orientation.z  = filtered_att_q.z();
	new_odom.twist.twist.linear.x  = 0.0;
	new_odom.twist.twist.linear.y  = 0.0;
	new_odom.twist.twist.linear.z  = 0.0;
	new_odom.twist.twist.angular.x = 0.0;
	new_odom.twist.twist.angular.y = 0.0;
	new_odom.twist.twist.angular.z = 0.0;
	
	// Update marker
	Eigen::Quaterniond q = (filtered_att_q * q_mesh).normalized();
	Eigen::Matrix3d R(filtered_att_q.toRotationMatrix());
	Eigen::Vector3d dp = R*pos_mesh;
	diff_marker.header.stamp = g_node->get_clock()->now();
	{
		auto clear_all = diff_marker;
		clear_all.header.stamp = g_node->get_clock()->now();
		clear_all.action = visualization_msgs::msg::Marker::DELETEALL;
		marker_pub->publish(clear_all);
	}
	diff_marker.pose.position.x = x - dp.x();
	diff_marker.pose.position.y = y - dp.y();
	diff_marker.pose.position.z = filtered_height_z - dp.z();
	diff_marker.pose.orientation.w = q.w();
	diff_marker.pose.orientation.x = q.x();
	diff_marker.pose.orientation.y = q.y();
	diff_marker.pose.orientation.z = q.z();
	
	odom_pub->publish(new_odom);
	marker_pub->publish(diff_marker);
}

void rcvTrajCallBack(const nav_msgs::msg::Path::SharedPtr traj) {
	nav_msgs::msg::Path path_msg;
	path_msg = *traj;
	geometry_msgs::msg::PoseStamped tmpPose;

	for(auto& pt: path_msg.poses){
		double x = pt.pose.position.x;
		double y = pt.pose.position.y;
		double yaw = pt.pose.position.z;
		getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
		pt.pose.position.z = terrain_to_body_height(now_p.z());
	}
	
	path_msg.header.frame_id = "map";
	path_msg.header.stamp = g_node->get_clock()->now();
	traj_pub->publish(path_msg);
}

void normyaw(double& y)
{
  if (y > M_PI)
  {
    y-=2*M_PI;
  }
  else if (y < -M_PI)
  {
    y+=2*M_PI;
  }
}

double calYawFromR(Eigen::Matrix3d R)
{
	Eigen::Vector2d p(R(0, 2), R(1, 2));
	Eigen::Vector2d b(R(0, 0), R(1, 0));
	Eigen::Vector2d x = (Eigen::Matrix2d::Identity()+p*p.transpose()/(1.0-p.squaredNorm()))*b;
	return atan2(x(1), x(0));
}
void normYaw(double& th)
{
	while (th > M_PI)
		th -= M_PI * 2;
	while (th < -M_PI)
		th += M_PI * 2;
}
void odomPubCallBack()
{
	// removed creation of temporary node to avoid duplicate node names
	if ((g_node->get_clock()->now() - begin_time).seconds() > 5.0)
		cloud_timer->cancel();

	if (require_initialpose && !pose_initialized)
	{
		new_odom.header.stamp = g_node->get_clock()->now();
		odom_pub->publish(new_odom);
		if (!marker_hidden_sent)
		{
			auto clear_marker = diff_marker;
			clear_marker.header.stamp = g_node->get_clock()->now();
			clear_marker.action = visualization_msgs::msg::Marker::DELETE;
			marker_pub->publish(clear_marker);
			marker_hidden_sent = true;
		}
		return;
	}

	if (!rcv_cmd)
	{
		new_odom.header.stamp = g_node->get_clock()->now();
    	odom_pub->publish(new_odom);
		diff_marker.header.stamp = g_node->get_clock()->now();
		marker_pub->publish(diff_marker);
		return;
	}

	double cmd_v = cmd_vel_msg.linear.x;
	double cmd_steer = cmd_vel_msg.angular.z;
	if (has_last_cmd_time && (g_node->get_clock()->now() - last_cmd_time).seconds() > cmd_timeout_sec)
	{
		cmd_v = 0.0;
		cmd_steer = 0.0;
	}
	if (!std::isfinite(cmd_v) || !std::isfinite(cmd_steer))
	{
		RCLCPP_WARN(g_node->get_logger(), "Received non-finite command, forcing zero command this cycle");
		cmd_v = 0.0;
		cmd_steer = 0.0;
	}

	const double target_v = std::clamp(cmd_v, min_cmd_speed, max_cmd_speed);
	double target_steer = std::clamp(cmd_steer, -max_cmd_steer, max_cmd_steer);

	if (enable_goal_watchdog && goal_active)
	{
		double dist_goal = std::hypot(goal_x - x, goal_y - y);
		auto now_t = g_node->get_clock()->now();
		if (!has_last_goal_dist)
		{
			last_goal_dist = dist_goal;
			has_last_goal_dist = true;
			last_goal_progress_time = now_t;
			has_goal_progress_time = true;
		}
		if (dist_goal + goal_progress_eps < last_goal_dist)
		{
			last_goal_dist = dist_goal;
			last_goal_progress_time = now_t;
		}
		else if (has_goal_progress_time && !goal_hold && dist_goal > goal_stop_radius)
		{
			if ((now_t - last_goal_progress_time).seconds() > goal_no_progress_sec)
			{
				goal_hold = true;
				RCLCPP_WARN(g_node->get_logger(),
					"Goal hold by no-progress watchdog: dist=%.2f, no progress for %.2fs",
					dist_goal, (now_t - last_goal_progress_time).seconds());
			}
		}
		if (dist_goal > goal_abort_radius)
		{
			goal_hold = true;
		}
		if (dist_goal < goal_stop_radius)
		{
			goal_hold = true;
		}
		else if (goal_hold && dist_goal > goal_release_radius)
		{
			goal_hold = false;
		}
	}

	double target_v_eff = target_v;
	if (enable_goal_watchdog && goal_hold)
	{
		target_v_eff = 0.0;
		target_steer = 0.0;
	}

	if (std::abs(v) < 0.08)
	{
		target_steer *= 0.5;
	}
	const double dv_max = std::max(0.0, max_cmd_accel) * time_resolution;
	const double ds_max = std::max(0.0, max_cmd_steer_rate) * time_resolution;
	v += std::clamp(target_v_eff - v, -dv_max, dv_max);
	steer += std::clamp(target_steer - steer, -ds_max, ds_max);

	const double steer_safe = std::clamp(steer, -1.2, 1.2);
	vx = v * cos(yaw);
	vy = v * sin(yaw);
	w = v * tan(steer_safe) / wheel_base;
	double x_next = x + vx * time_resolution;
	double y_next = y + vy * time_resolution;
	double yaw_next = yaw + w * time_resolution;
	if (!std::isfinite(x_next) || !std::isfinite(y_next) || !std::isfinite(yaw_next))
	{
		RCLCPP_WARN(g_node->get_logger(), "State update became non-finite, skipping this cycle");
		return;
	}
	double step = std::hypot(x_next - x, y_next - y);
	if (step > std::max(0.005, max_position_step))
	{
		double scale = max_position_step / step;
		x_next = x + (x_next - x) * scale;
		y_next = y + (y_next - y) * scale;
	}
	if (has_map_bounds)
	{
		const double min_x = map_min_x - map_bound_margin;
		const double max_x = map_max_x + map_bound_margin;
		const double min_y = map_min_y - map_bound_margin;
		const double max_y = map_max_y + map_bound_margin;
		if (x_next < min_x || x_next > max_x || y_next < min_y || y_next > max_y)
		{
			x_next = std::clamp(x_next, min_x, max_x);
			y_next = std::clamp(y_next, min_y, max_y);
			v = 0.0;
			steer = 0.0;
			goal_hold = true;
		}
	}
	double dyaw_step = yaw_next - yaw;
	if (dyaw_step > max_yaw_step) dyaw_step = max_yaw_step;
	if (dyaw_step < -max_yaw_step) dyaw_step = -max_yaw_step;

	x = x_next;
	y = y_next;
	yaw = yaw + dyaw_step;
	normYaw(yaw);
	new_odom.header.stamp = g_node->get_clock()->now();
	new_odom.pose.pose.position.x  = x;
	new_odom.pose.pose.position.y  = y;

	/*hxlin debug*/
	getSE3(Eigen::Vector3d(x, y, yaw), now_p, now_R);
	const double update_z = terrain_to_body_height(now_p.z());
	if (!has_filtered_height)
	{
		filtered_height_z = update_z;
		has_filtered_height = true;
	}
	else
	{
		double delta_z = update_z - filtered_height_z;
		if (std::abs(delta_z) < z_deadband)
		{
			delta_z = 0.0;
		}
		const double max_step = std::max(0.0, z_max_rate) * time_resolution;
		delta_z = std::clamp(delta_z, -max_step, max_step);
		filtered_height_z += delta_z;
	}
	new_odom.pose.pose.position.z  = filtered_height_z;


	Eigen::Quaterniond q_body(now_R);
	q_body.normalize();
	if (!has_filtered_att)
	{
		filtered_att_q = q_body;
		has_filtered_att = true;
	}
	else
	{
		filtered_att_q = filtered_att_q.slerp(std::clamp(att_filter_alpha, 0.01, 1.0), q_body);
		filtered_att_q.normalize();
	}
	new_odom.pose.pose.orientation.w  = filtered_att_q.w();
	new_odom.pose.pose.orientation.x  = filtered_att_q.x();
	new_odom.pose.pose.orientation.y  = filtered_att_q.y();
	new_odom.pose.pose.orientation.z  = filtered_att_q.z();
	new_odom.twist.twist.linear.x  = vx;
	new_odom.twist.twist.linear.y  = vy;
	new_odom.twist.twist.linear.z  = 0;
	new_odom.twist.twist.angular.x = v;
	new_odom.twist.twist.angular.y = steer;
	new_odom.twist.twist.angular.z = w;

	Eigen::Quaterniond q = (filtered_att_q * q_mesh).normalized();
	Eigen::Matrix3d R(filtered_att_q.toRotationMatrix());
	Eigen::Vector3d dp = R*pos_mesh;
	diff_marker.header.stamp = g_node->get_clock()->now();
	diff_marker.pose.position.x = x - dp.x();
	diff_marker.pose.position.y = y - dp.y();
	diff_marker.pose.position.z = filtered_height_z - dp.z();
	diff_marker.pose.orientation.w = q.w();
	diff_marker.pose.orientation.x = q.x();
	diff_marker.pose.orientation.y = q.y();
	diff_marker.pose.orientation.z = q.z();
	
	odom_pub->publish(new_odom);
	marker_pub->publish(diff_marker);
}

void cloudCallBack()
{
	cloud_pub->publish(world_cloud_msg);
}

int main (int argc, char** argv) 
{        
    rclcpp::init (argc, argv);
    g_node = std::make_shared<rclcpp::Node>("ugv_kinematic_model_node");
    auto nh = g_node;
	
	/* Hxlin Debug */
    nh->declare_parameter<double>("max_height", 1.0);
	nh->declare_parameter<bool>("require_initialpose", true);
	nh->declare_parameter<bool>("allow_runtime_reinit", true);
	nh->declare_parameter<double>("reinit_min_interval", 0.5);
	nh->declare_parameter<double>("reinit_max_move_when_running", 2.0);
	nh->declare_parameter<double>("max_position_step", 0.08);
	nh->declare_parameter<double>("max_yaw_step", 0.25);
	nh->declare_parameter<double>("z_deadband", 0.01);
	nh->declare_parameter<double>("z_max_rate", 0.5);
	nh->declare_parameter<double>("max_cmd_speed", 1.5);
	nh->declare_parameter<double>("min_cmd_speed", -0.8);
	nh->declare_parameter<double>("max_cmd_accel", 1.0);
	nh->declare_parameter<double>("max_cmd_steer", 0.6);
	nh->declare_parameter<double>("max_cmd_steer_rate", 1.0);
	nh->declare_parameter<double>("wheel_base", 0.6);
	nh->declare_parameter<bool>("enable_goal_watchdog", true);
	nh->declare_parameter<double>("goal_stop_radius", 0.35);
	nh->declare_parameter<double>("goal_release_radius", 0.60);
	nh->declare_parameter<double>("goal_abort_radius", 25.0);
	nh->declare_parameter<double>("goal_no_progress_sec", 3.0);
	nh->declare_parameter<double>("goal_progress_eps", 0.03);
	nh->declare_parameter<double>("cmd_timeout_sec", 0.4);
	nh->declare_parameter<double>("map_bound_margin", 0.3);
	nh->declare_parameter<double>("att_filter_alpha", 0.18);
    nh->declare_parameter<double>("init_x", 1.0);
    nh->declare_parameter<double>("init_y", 1.0);
    nh->declare_parameter<double>("init_z", 1.0);
	nh->declare_parameter<double>("init_yaw", 0.0);
	nh->declare_parameter<std::string>("map_name", "quick.pcd");
    
    max_height = nh->get_parameter("max_height").as_double();
	require_initialpose = nh->get_parameter("require_initialpose").as_bool();
	allow_runtime_reinit = nh->get_parameter("allow_runtime_reinit").as_bool();
	reinit_min_interval = nh->get_parameter("reinit_min_interval").as_double();
	reinit_max_move_when_running = nh->get_parameter("reinit_max_move_when_running").as_double();
	max_position_step = nh->get_parameter("max_position_step").as_double();
	max_yaw_step = nh->get_parameter("max_yaw_step").as_double();
	z_deadband = nh->get_parameter("z_deadband").as_double();
	z_max_rate = nh->get_parameter("z_max_rate").as_double();
	max_cmd_speed = nh->get_parameter("max_cmd_speed").as_double();
	min_cmd_speed = nh->get_parameter("min_cmd_speed").as_double();
	max_cmd_accel = nh->get_parameter("max_cmd_accel").as_double();
	max_cmd_steer = nh->get_parameter("max_cmd_steer").as_double();
	max_cmd_steer_rate = nh->get_parameter("max_cmd_steer_rate").as_double();
	wheel_base = nh->get_parameter("wheel_base").as_double();
	enable_goal_watchdog = nh->get_parameter("enable_goal_watchdog").as_bool();
	goal_stop_radius = nh->get_parameter("goal_stop_radius").as_double();
	goal_release_radius = nh->get_parameter("goal_release_radius").as_double();
	goal_abort_radius = nh->get_parameter("goal_abort_radius").as_double();
	goal_no_progress_sec = nh->get_parameter("goal_no_progress_sec").as_double();
	goal_progress_eps = nh->get_parameter("goal_progress_eps").as_double();
	cmd_timeout_sec = nh->get_parameter("cmd_timeout_sec").as_double();
	map_bound_margin = nh->get_parameter("map_bound_margin").as_double();
	att_filter_alpha = nh->get_parameter("att_filter_alpha").as_double();
    p_init_x = nh->get_parameter("init_x").as_double();
    p_init_y = nh->get_parameter("init_y").as_double();
    p_init_z = nh->get_parameter("init_z").as_double();
	yaw_init = nh->get_parameter("init_yaw").as_double();
	map_name = nh->get_parameter("map_name").as_string();
	if (!std::isfinite(wheel_base) || wheel_base <= 1e-3)
	{
		RCLCPP_WARN(nh->get_logger(), "Invalid wheel_base %.3f, fallback to 0.6", wheel_base);
		wheel_base = 0.6;
	}
	pose_initialized = !require_initialpose;

	initParams();

	traj_sub = nh->create_subscription<nav_msgs::msg::Path>( 
        "/visualization/refinedTraj_kinoastar", 1, rcvTrajCallBack );
	cmd_vel_sub = nh->create_subscription<geometry_msgs::msg::Twist>( 
        "cmd_vel", 1, rcvVelCmdCallBack );
	initialpose_sub = nh->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "initialpose", 10, rcvInitialPoseCallBack );
	if (enable_goal_watchdog)
	{
		goal_sub = nh->create_subscription<geometry_msgs::msg::PoseStamped>(
			"/move_base_simple/goal", 10, rcvGoalCallBack );
		RCLCPP_INFO(nh->get_logger(), "Simulator goal watchdog enabled on /move_base_simple/goal");
	}
	else
	{
		RCLCPP_INFO(nh->get_logger(), "Simulator goal watchdog disabled; cmd_vel is not gated by goal hold");
	}
    odom_pub  = nh->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
	traj_pub = nh->create_publisher<nav_msgs::msg::Path>("traj_has_height", 1);
	marker_pub = nh->create_publisher<visualization_msgs::msg::Marker>("marker", 1);
    cloud_pub  = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/global_cloud", 1);
	cloud_timer = nh->create_wall_timer(
        std::chrono::seconds(2), cloudCallBack);
	rclcpp::TimerBase::SharedPtr odom_timer = nh->create_wall_timer(
        std::chrono::milliseconds((int)(time_resolution * 1000)), odomPubCallBack);
	begin_time = nh->get_clock()->now();
	Eigen::Matrix3d R_mesh;
	R_mesh << 1, 0, 0,
			  0, 1, 0,
			  0, 0, 1;
	q_mesh = Eigen::Quaterniond{R_mesh};
	pos_mesh = Eigen::Vector3d(0, 0, 0.0);
	{
		auto clear_all = diff_marker;
		clear_all.header.stamp = g_node->get_clock()->now();
		clear_all.action = visualization_msgs::msg::Marker::DELETEALL;
		marker_pub->publish(clear_all);
	}
	rclcpp::spin(nh);
    return 0;
}