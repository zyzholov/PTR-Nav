#include <rclcpp/rclcpp.hpp>
#include <gazebo_ros/node.hpp>
#include <rclcpp/logging.hpp>

#include <gazebo/physics/Model.hh>
#include <gazebo/physics/MultiRayShape.hh>// Store the latest laser scans into laserMsg
#include <gazebo/physics/PhysicsEngine.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <gazebo/transport/Node.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include "ros2_livox/livox_points_plugin.h"
#include "ros2_livox/csv_reader.hpp"
#include "ros2_livox/livox_ode_multiray_shape.h"
#include <livox_ros_driver2/msg/custom_msg.hpp>

namespace gazebo
{

    GZ_REGISTER_SENSOR_PLUGIN(LivoxPointsPlugin)

    LivoxPointsPlugin::LivoxPointsPlugin() {}

    LivoxPointsPlugin::~LivoxPointsPlugin() {}

	    void convertDataToRotateInfo(const std::vector<std::vector<double>> &datas, std::vector<AviaRotateInfo> &avia_infos)
    {
        avia_infos.reserve(datas.size());
        double deg_2_rad = M_PI / 180.0;
        for (auto &data : datas)
        {
            if (data.size() == 3)
            {
                avia_infos.emplace_back();
                avia_infos.back().time = data[0];
                avia_infos.back().azimuth = data[1] * deg_2_rad;
                avia_infos.back().zenith = data[2] * deg_2_rad - M_PI_2; //转化成标准的右手系角度
            }
            else
            {
                RCLCPP_ERROR(rclcpp::get_logger("convertDataToRotateInfo"), "data size is not 3!");
            }
        }
    }

	    std::vector<double> ParseScalarList(const std::string &text)
	    {
	        std::vector<double> values;
	        std::istringstream stream(text);
	        double value = 0.0;
	        while (stream >> value)
	        {
	            values.push_back(value);
	        }
	        return values;
	    }

	    void CollectTerrainCylinders(sdf::ElementPtr elem, std::vector<TerrainCylinder> &cylinders)
	    {
	        if (!elem)
	        {
	            return;
	        }

	        if (elem->GetName() == "collision" && elem->HasElement("geometry"))
	        {
	            auto geometry = elem->GetElement("geometry");
	            if (geometry && geometry->HasElement("cylinder"))
	            {
	                auto cylinder = geometry->GetElement("cylinder");
	                TerrainCylinder out;
	                out.radius = cylinder->HasElement("radius") ? cylinder->Get<double>("radius") : 0.0;
	                out.half_length = 0.5 * (cylinder->HasElement("length") ? cylinder->Get<double>("length") : 0.0);
	                if (elem->HasElement("pose"))
	                {
	                    const auto values = ParseScalarList(elem->Get<std::string>("pose"));
	                    if (values.size() >= 3)
	                    {
	                        out.center.Set(values[0], values[1], values[2]);
	                    }
	                }
	                if (out.radius > 0.0 && out.half_length > 0.0)
	                {
	                    cylinders.push_back(out);
	                }
	            }
	        }

	        for (auto child = elem->GetFirstElement(); child; child = child->GetNextElement())
	        {
	            CollectTerrainCylinders(child, cylinders);
	        }
	    }

    uint64_t SimTimeToNanoseconds(const gazebo::common::Time &time)
    {
        return static_cast<uint64_t>(time.sec) * 1000000000ULL +
               static_cast<uint64_t>(time.nsec);
    }

    builtin_interfaces::msg::Time SimTimeToRosStamp(const gazebo::common::Time &time)
    {
        builtin_interfaces::msg::Time stamp;
        stamp.sec = static_cast<int32_t>(time.sec);
        stamp.nanosec = static_cast<uint32_t>(time.nsec);
        return stamp;
    }

    void LivoxPointsPlugin::Load(gazebo::sensors::SensorPtr _parent, sdf::ElementPtr sdf)
    {
        node_ = gazebo_ros::Node::Get(sdf);
        
        std::vector<std::vector<double>> datas;
        std::string file_name = sdf->Get<std::string>("csv_file_name");
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "load csv file name: %s", file_name.c_str());
        if (!CsvReader::ReadCsvFile(file_name, datas))
        {   
            RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "cannot get csv file! %s will return !", file_name.c_str());
            return;
        }
        sdfPtr = sdf;
        auto rayElem = sdfPtr->GetElement("ray");
        auto scanElem = rayElem->GetElement("scan");
        auto rangeElem = rayElem->GetElement("range");


        raySensor = _parent;
        auto sensor_pose = raySensor->Pose();
        // Keep the Gazebo ray sensor actively updating even when no Gazebo-side
        // subscriber is attached; otherwise the ROS topics appear but never emit.
        raySensor->SetActive(true);
        auto curr_scan_topic = sdf->Get<std::string>("topic");
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "ros topic name: %s", curr_scan_topic.c_str());

        child_name = raySensor->Name();
        parent_name = raySensor->ParentName();
        size_t delimiter_pos = parent_name.find("::");
        parent_name = parent_name.substr(delimiter_pos + 2);

        node = transport::NodePtr(new transport::Node());
        node->Init(raySensor->WorldName());
        // PointCloud2 publisher
        cloud2_pub = node_->create_publisher<sensor_msgs::msg::PointCloud2>(curr_scan_topic + "_PointCloud2", 10);
        // CustomMsg publisher
        custom_pub = node_->create_publisher<livox_ros_driver2::msg::CustomMsg>(curr_scan_topic, 10);

        scanPub = node->Advertise<msgs::LaserScanStamped>(curr_scan_topic+"laserscan", 50);

        aviaInfos.clear();
        convertDataToRotateInfo(datas, aviaInfos);
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "scan info size: %ld", aviaInfos.size());
        maxPointSize = aviaInfos.size();

        RayPlugin::Load(_parent, sdfPtr);
        laserMsg.mutable_scan()->set_frame(_parent->ParentName());
        // parentEntity = world->GetEntity(_parent->ParentName());
        parentEntity = this->world->EntityByName(_parent->ParentName());
        //SendRosTf(sensor_pose, raySensor->ParentName(), raySensor->Name());
        auto physics = world->Physics();
        laserCollision = physics->CreateCollision("multiray", _parent->ParentName());
        laserCollision->SetName("ray_sensor_collision");
        laserCollision->SetRelativePose(_parent->Pose());
        laserCollision->SetInitialRelativePose(_parent->Pose());
        rayShape.reset(new gazebo::physics::LivoxOdeMultiRayShape(laserCollision));
        laserCollision->SetShape(rayShape);
        samplesStep = sdfPtr->Get<int>("samples");
        downSample = sdfPtr->Get<int>("downsample");
        if (downSample < 1)
        {
            downSample = 1;
        }
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "sample: %ld", samplesStep);
        RCLCPP_INFO(rclcpp::get_logger("LivoxPointsPlugin"), "downsample: %ld", downSample);
        rayShape->RayShapes().reserve(samplesStep / downSample);
        rayShape->Load(sdfPtr);
        rayShape->Init();
        minDist = rangeElem->Get<double>("min");
        maxDist = rangeElem->Get<double>("max");
        self_filter_enable = sdfPtr->HasElement("self_filter_enable") ?
            sdfPtr->Get<bool>("self_filter_enable") : false;
        self_filter_sensor_x = sdfPtr->HasElement("self_filter_sensor_x") ?
            sdfPtr->Get<double>("self_filter_sensor_x") : 0.0;
        self_filter_sensor_y = sdfPtr->HasElement("self_filter_sensor_y") ?
            sdfPtr->Get<double>("self_filter_sensor_y") : 0.0;
        self_filter_sensor_z = sdfPtr->HasElement("self_filter_sensor_z") ?
            sdfPtr->Get<double>("self_filter_sensor_z") : 0.0;
        self_filter_sensor_yaw = sdfPtr->HasElement("self_filter_sensor_yaw") ?
            sdfPtr->Get<double>("self_filter_sensor_yaw") : 0.0;
        self_filter_min_x = sdfPtr->HasElement("self_filter_min_x") ?
            sdfPtr->Get<double>("self_filter_min_x") : -0.5;
        self_filter_max_x = sdfPtr->HasElement("self_filter_max_x") ?
            sdfPtr->Get<double>("self_filter_max_x") : 0.5;
        self_filter_min_y = sdfPtr->HasElement("self_filter_min_y") ?
            sdfPtr->Get<double>("self_filter_min_y") : -0.3;
        self_filter_max_y = sdfPtr->HasElement("self_filter_max_y") ?
            sdfPtr->Get<double>("self_filter_max_y") : 0.3;
        self_filter_min_z = sdfPtr->HasElement("self_filter_min_z") ?
            sdfPtr->Get<double>("self_filter_min_z") : -0.3;
        self_filter_max_z = sdfPtr->HasElement("self_filter_max_z") ?
            sdfPtr->Get<double>("self_filter_max_z") : 0.2;
        terrain_mode = sdfPtr->HasElement("terrain_mode") ?
            sdfPtr->Get<bool>("terrain_mode") : false;
        if (terrain_mode)
        {
            const auto heightmap_file = sdfPtr->HasElement("terrain_heightmap") ?
                sdfPtr->Get<std::string>("terrain_heightmap") : std::string();
            terrain_size_x = sdfPtr->HasElement("terrain_size_x") ?
                sdfPtr->Get<double>("terrain_size_x") : terrain_size_x;
            terrain_size_y = sdfPtr->HasElement("terrain_size_y") ?
                sdfPtr->Get<double>("terrain_size_y") : terrain_size_y;
            terrain_size_z = sdfPtr->HasElement("terrain_size_z") ?
                sdfPtr->Get<double>("terrain_size_z") : terrain_size_z;
            terrain_origin_x = sdfPtr->HasElement("terrain_origin_x") ?
                sdfPtr->Get<double>("terrain_origin_x") : terrain_origin_x;
            terrain_origin_y = sdfPtr->HasElement("terrain_origin_y") ?
                sdfPtr->Get<double>("terrain_origin_y") : terrain_origin_y;
            terrain_origin_z = sdfPtr->HasElement("terrain_origin_z") ?
                sdfPtr->Get<double>("terrain_origin_z") : terrain_origin_z;
            terrain_ray_step = std::max(0.10, sdfPtr->HasElement("terrain_ray_step") ?
                sdfPtr->Get<double>("terrain_ray_step") : terrain_ray_step);
            if (!heightmap_file.empty() && terrain_image.Load(heightmap_file) == 0 && terrain_image.Valid())
            {
                terrain_image_valid = true;
                terrain_image_width = terrain_image.GetWidth();
                terrain_image_height = terrain_image.GetHeight();
                terrain_heights.assign(
                    static_cast<size_t>(terrain_image_width) * static_cast<size_t>(terrain_image_height), 0.0);
                for (unsigned int py = 0; py < terrain_image_height; ++py)
                {
                    for (unsigned int px = 0; px < terrain_image_width; ++px)
                    {
                        const ignition::math::Color c = terrain_image.Pixel(px, py);
                        terrain_heights[static_cast<size_t>(py) * terrain_image_width + px] =
                            terrain_origin_z + std::clamp(c.R(), 0.0f, 1.0f) * terrain_size_z;
                    }
                }
                RCLCPP_INFO(
                    rclcpp::get_logger("LivoxPointsPlugin"),
                    "terrain heightmap raycast mode enabled: %s (%ux%u), step=%.3f, samples=%ld/%ld",
                    heightmap_file.c_str(), terrain_image_width, terrain_image_height,
                    terrain_ray_step, samplesStep, downSample);
            }
            else
            {
                RCLCPP_ERROR(
                    rclcpp::get_logger("LivoxPointsPlugin"),
                    "terrain heightmap mode requested but heightmap failed to load: %s",
                    heightmap_file.c_str());
            }

            const std::string model_sdf_file =
                heightmap_file.find("/materials/textures/") == std::string::npos ?
                std::string() : heightmap_file.substr(0, heightmap_file.find("/materials/textures/")) + "/model.sdf";
            if (!model_sdf_file.empty())
            {
                sdf::SDFPtr terrain_sdf = sdf::readFile(model_sdf_file);
                if (terrain_sdf && terrain_sdf->Root())
                {
                    CollectTerrainCylinders(terrain_sdf->Root(), terrain_cylinders);
                    RCLCPP_INFO(
                        rclcpp::get_logger("LivoxPointsPlugin"),
                        "terrain fast raycast loaded %zu cylinder obstacles from %s",
                        terrain_cylinders.size(), model_sdf_file.c_str());
                }
            }
        }
        if (self_filter_enable)
        {
            RCLCPP_INFO(
                rclcpp::get_logger("LivoxPointsPlugin"),
                "self filter enabled: base box x[%.2f, %.2f] y[%.2f, %.2f] z[%.2f, %.2f]",
                self_filter_min_x, self_filter_max_x,
                self_filter_min_y, self_filter_max_y,
                self_filter_min_z, self_filter_max_z);
        }
        auto offset = laserCollision->RelativePose();
        ignition::math::Vector3d start_point, end_point;
        for (int j = 0; j < samplesStep; j += downSample)
        {
            int index = j % maxPointSize;
            auto &rotate_info = aviaInfos[index];
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            auto axis = offset.Rot() * ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            start_point = minDist * axis + offset.Pos();
            end_point = maxDist * axis + offset.Pos();
            rayShape->AddRay(start_point, end_point);
        }
    }

    void LivoxPointsPlugin::OnNewLaserScans()
{
    // Check if rayShape has been initialized
    if (rayShape)
    {
        const bool publish_scan = scanPub && scanPub->HasConnections();
        const bool publish_custom = custom_pub && custom_pub->get_subscription_count() > 0;
        const bool publish_cloud2 = cloud2_pub && cloud2_pub->get_subscription_count() > 0;
        if (!publish_scan && !publish_custom && !publish_cloud2)
        {
            return;
        }

        std::vector<std::pair<int, AviaRotateInfo>> points_pair;
        // Initialize ray scan point pairs
        InitializeRays(points_pair, rayShape);
        rayShape->Update();

        const auto sim_time = world->SimTime();
        const uint64_t scan_stamp_ns = SimTimeToNanoseconds(sim_time);
        const builtin_interfaces::msg::Time scan_stamp = SimTimeToRosStamp(sim_time);

        // Create laser scan message and set the timestamp
        msgs::Set(laserMsg.mutable_time(), world->SimTime());
        msgs::LaserScan *scan = laserMsg.mutable_scan();
        InitializeScan(scan);

        // Create a custom message pp_livox for publishing Livox CustomMsg type messages
        livox_ros_driver2::msg::CustomMsg pp_livox;
        if (publish_custom)
        {
            pp_livox.header.stamp = scan_stamp;
            pp_livox.header.frame_id = raySensor->Name();
            pp_livox.timebase = scan_stamp_ns;
            pp_livox.lidar_id = 0;
            pp_livox.points.reserve(points_pair.size());
        }
        int count = 0;

        // For publishing PointCloud2 type messages
        sensor_msgs::msg::PointCloud cloud;
        if (publish_cloud2)
        {
            cloud.header.stamp = scan_stamp;
            cloud.header.frame_id = raySensor->Name();
            cloud.points.reserve(points_pair.size());
        }
        auto &clouds = cloud.points;

        // Iterate over ray scan point pairs
        for (auto &pair : points_pair)
        {
            auto range = rayShape->GetRange(pair.first);
            auto intensity = rayShape->GetRetro(pair.first);

            // Handle out-of-range data
            bool valid_range = true;
            if (range >= RangeMax())
            {
                range = 0;
                valid_range = false;
            }
            else if (range <= RangeMin())
            {
                range = 0;
                valid_range = false;
            }

            // Calculate point cloud data
            auto rotate_info = pair.second;
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            auto axis = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            auto point = range * axis;

            if (valid_range && IsSelfFiltered(point))
            {
                count++;
                continue;
            }

            if (publish_custom)
            {
                livox_ros_driver2::msg::CustomPoint p;
                p.x = point.X();
                p.y = point.Y();
                p.z = point.Z();
                p.reflectivity = static_cast<uint8_t>(std::clamp(intensity, 0.0, 255.0));
                p.tag = 0x10;
                p.line = static_cast<uint8_t>(count % 4);
                p.offset_time = 0;
                pp_livox.points.push_back(p);
            }

            if (publish_cloud2)
            {
                clouds.emplace_back();
                clouds.back().x = point.X();
                clouds.back().y = point.Y();
                clouds.back().z = point.Z();
            }
            count++;
        }

        if (publish_scan) scanPub->Publish(laserMsg);

        // Set the number of point cloud data and publish the CustomMsg message
        if (publish_custom)
        {
            pp_livox.point_num = static_cast<uint32_t>(pp_livox.points.size());
            custom_pub->publish(pp_livox);
        }

        // Publish PointCloud2 type message
        if (publish_cloud2)
        {
            sensor_msgs::msg::PointCloud2 cloud2;
            sensor_msgs::convertPointCloudToPointCloud2(cloud, cloud2);
            cloud2.header = cloud.header;
            cloud2_pub->publish(cloud2);
        }
    }
}

    bool LivoxPointsPlugin::TerrainHeightAt(double world_x, double world_y, double &world_z) const
    {
        if (!terrain_image_valid || terrain_image_width < 2 || terrain_image_height < 2)
        {
            return false;
        }

        const double u = (world_x - (terrain_origin_x - terrain_size_x * 0.5)) / terrain_size_x;
        const double v = (world_y - (terrain_origin_y - terrain_size_y * 0.5)) / terrain_size_y;
        if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
        {
            return false;
        }

        const auto px = static_cast<unsigned int>(std::clamp(
            std::llround(u * static_cast<double>(terrain_image_width - 1)),
            0LL, static_cast<long long>(terrain_image_width - 1)));
        // Gazebo heightmaps use image row 0 as the positive-y side.
        const auto py = static_cast<unsigned int>(std::clamp(
            std::llround((1.0 - v) * static_cast<double>(terrain_image_height - 1)),
            0LL, static_cast<long long>(terrain_image_height - 1)));
        const auto idx = static_cast<size_t>(py) * terrain_image_width + px;
        if (idx >= terrain_heights.size())
        {
            return false;
        }
        world_z = terrain_heights[idx];
        return true;
    }

    bool LivoxPointsPlugin::TerrainRayIntersection(
        const ignition::math::Vector3d &sensor_world,
        const ignition::math::Vector3d &axis_world,
        ignition::math::Vector3d &hit_world) const
    {
        if (!terrain_image_valid)
        {
            return false;
        }

        if (axis_world.Length() < 1e-6 || axis_world.Z() >= -1e-4)
        {
            return false;
        }

        const double x_min = terrain_origin_x - terrain_size_x * 0.5;
        const double x_max = terrain_origin_x + terrain_size_x * 0.5;
        const double y_min = terrain_origin_y - terrain_size_y * 0.5;
        const double y_max = terrain_origin_y + terrain_size_y * 0.5;
        double t_enter = RangeMin();
        double t_exit = RangeMax();
        auto clip_axis = [&](double origin, double dir, double min_v, double max_v) -> bool {
            if (std::abs(dir) < 1e-9)
            {
                return origin >= min_v && origin <= max_v;
            }
            double t0 = (min_v - origin) / dir;
            double t1 = (max_v - origin) / dir;
            if (t0 > t1)
                std::swap(t0, t1);
            t_enter = std::max(t_enter, t0);
            t_exit = std::min(t_exit, t1);
            return t_enter <= t_exit;
        };
        if (!clip_axis(sensor_world.X(), axis_world.X(), x_min, x_max) ||
            !clip_axis(sensor_world.Y(), axis_world.Y(), y_min, y_max))
        {
            return false;
        }

        const double terrain_top = terrain_origin_z + terrain_size_z;
        const double terrain_bottom = terrain_origin_z;
        if (sensor_world.Z() < terrain_bottom && axis_world.Z() <= 0.0)
        {
            return false;
        }
        if (sensor_world.Z() > terrain_top)
        {
            t_enter = std::max(t_enter, (terrain_top - sensor_world.Z()) / axis_world.Z());
        }
        t_exit = std::min(t_exit, (terrain_bottom - sensor_world.Z()) / axis_world.Z());
        if (t_enter > t_exit || t_exit <= RangeMin())
        {
            return false;
        }

        double prev_dist = std::max(RangeMin(), t_enter);
        ignition::math::Vector3d prev_point = sensor_world + axis_world * prev_dist;
        double prev_height = 0.0;
        bool prev_has_height = TerrainHeightAt(prev_point.X(), prev_point.Y(), prev_height);
        double prev_delta = prev_has_height ? (prev_point.Z() - prev_height) : 1.0;
        if (prev_has_height && prev_delta <= 0.0)
        {
            hit_world = prev_point;
            hit_world.Z() = prev_height;
            return true;
        }

        for (double dist = prev_dist + terrain_ray_step; dist <= t_exit + 1e-6; dist += terrain_ray_step)
        {
            if (dist > t_exit)
            {
                dist = t_exit;
            }
            const ignition::math::Vector3d point = sensor_world + axis_world * dist;
            double height = 0.0;
            if (!TerrainHeightAt(point.X(), point.Y(), height))
            {
                prev_dist = dist;
                prev_point = point;
                prev_has_height = false;
                prev_delta = 1.0;
                continue;
            }

            const double delta = point.Z() - height;
            if (prev_has_height && prev_delta >= 0.0 && delta <= 0.0)
            {
                double lo = prev_dist;
                double hi = dist;
                for (int i = 0; i < 8; ++i)
                {
                    const double mid = 0.5 * (lo + hi);
                    const ignition::math::Vector3d mid_point = sensor_world + axis_world * mid;
                    double mid_height = 0.0;
                    if (!TerrainHeightAt(mid_point.X(), mid_point.Y(), mid_height))
                    {
                        lo = mid;
                        continue;
                    }
                    const double mid_delta = mid_point.Z() - mid_height;
                    if (mid_delta > 0.0)
                        lo = mid;
                    else
                        hi = mid;
                }
                const double hit_dist = 0.5 * (lo + hi);
                hit_world = sensor_world + axis_world * hit_dist;
                return true;
            }

            prev_dist = dist;
            prev_point = point;
            prev_has_height = true;
            prev_delta = delta;
        }

        return false;
    }

    bool LivoxPointsPlugin::TerrainCylinderIntersection(
        const ignition::math::Vector3d &sensor_world,
        const ignition::math::Vector3d &axis_world,
        double &hit_dist) const
    {
        bool hit = false;
        double best_dist = RangeMax();
        for (const auto &cyl : terrain_cylinders)
        {
            const double ox = sensor_world.X() - cyl.center.X();
            const double oy = sensor_world.Y() - cyl.center.Y();
            const double dx = axis_world.X();
            const double dy = axis_world.Y();
            const double a = dx * dx + dy * dy;
            if (a < 1e-9)
            {
                continue;
            }
            const double b = 2.0 * (ox * dx + oy * dy);
            const double c = ox * ox + oy * oy - cyl.radius * cyl.radius;
            const double disc = b * b - 4.0 * a * c;
            if (disc < 0.0)
            {
                continue;
            }
            const double sqrt_disc = std::sqrt(disc);
            const double inv = 0.5 / a;
            const double roots[2] = {(-b - sqrt_disc) * inv, (-b + sqrt_disc) * inv};
            for (const double dist : roots)
            {
                if (dist <= RangeMin() || dist >= best_dist || dist >= RangeMax())
                {
                    continue;
                }
                const double z = sensor_world.Z() + axis_world.Z() * dist;
                if (z >= cyl.center.Z() - cyl.half_length && z <= cyl.center.Z() + cyl.half_length)
                {
                    best_dist = dist;
                    hit = true;
                }
            }
        }

        if (hit)
        {
            hit_dist = best_dist;
        }
        return hit;
    }

    void LivoxPointsPlugin::PublishTerrainHeightmapRays()
    {
        const bool publish_custom = custom_pub && custom_pub->get_subscription_count() > 0;
        const bool publish_cloud2 = cloud2_pub && cloud2_pub->get_subscription_count() > 0;
        if (!publish_custom && !publish_cloud2)
        {
            return;
        }

        const double scan_rate_hz = std::max(1.0, raySensor ? raySensor->UpdateRate() : 5.0);
        const uint64_t scan_period_ns = static_cast<uint64_t>(std::llround(1e9 / scan_rate_hz));
        const uint64_t scan_stamp_ns = static_cast<uint64_t>(node_->get_clock()->now().nanoseconds());
        builtin_interfaces::msg::Time scan_stamp;
        scan_stamp.sec = static_cast<int32_t>(scan_stamp_ns / 1000000000ULL);
        scan_stamp.nanosec = static_cast<uint32_t>(scan_stamp_ns % 1000000000ULL);

        livox_ros_driver2::msg::CustomMsg pp_livox;
        if (publish_custom)
        {
            pp_livox.header.stamp = scan_stamp;
            pp_livox.header.frame_id = raySensor->Name();
            pp_livox.timebase = scan_stamp_ns;
            pp_livox.lidar_id = 0;
            pp_livox.points.reserve(static_cast<size_t>(std::max<int64_t>(1, samplesStep / downSample)));
        }

        sensor_msgs::msg::PointCloud cloud;
        if (publish_cloud2)
        {
            cloud.header.stamp = scan_stamp;
            cloud.header.frame_id = raySensor->Name();
            cloud.points.reserve(static_cast<size_t>(std::max<int64_t>(1, samplesStep / downSample)));
        }

        const ignition::math::Pose3d sensor_world =
            parentEntity ? (parentEntity->WorldPose() * raySensor->Pose()) : raySensor->Pose();
        const auto sensor_world_rot_inv = sensor_world.Rot().Inverse();
        const auto sensor_pos = sensor_world.Pos();

        std::vector<AviaRotateInfo> scan_infos;
        std::vector<double> unwrapped_pattern_time;
        const auto reserve_count = static_cast<size_t>(std::max<int64_t>(1, samplesStep / downSample));
        scan_infos.reserve(reserve_count);
        unwrapped_pattern_time.reserve(reserve_count);
        double wrap_offset = 0.0;
        double prev_time = 0.0;
        bool first_time = true;

        for (int k = currStartIndex; k < currStartIndex + samplesStep; k += downSample)
        {
            const int index = k % maxPointSize;
            const auto &rotate_info = aviaInfos[index];
            if (!first_time && rotate_info.time < prev_time)
            {
                wrap_offset += static_cast<double>(std::max<int64_t>(1, maxPointSize));
            }
            first_time = false;
            prev_time = rotate_info.time;
            scan_infos.push_back(rotate_info);
            unwrapped_pattern_time.push_back(rotate_info.time + wrap_offset);
        }
        currStartIndex += samplesStep;

        const double pattern_span = unwrapped_pattern_time.size() > 1 ?
            std::max(1e-6, unwrapped_pattern_time.back() - unwrapped_pattern_time.front()) : 1.0;
        const auto total_candidates = std::max<size_t>(1, scan_infos.size());

        for (size_t count = 0; count < scan_infos.size(); ++count)
        {
            const auto &rotate_info = scan_infos[count];
            ignition::math::Quaterniond ray;
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            const auto axis_sensor = ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            auto axis_world = sensor_world.Rot().RotateVector(axis_sensor);
            axis_world.Normalize();

            ignition::math::Vector3d world_point;
            bool has_hit = false;
            double hit_range = RangeMax();
            if (TerrainRayIntersection(sensor_pos, axis_world, world_point))
            {
                hit_range = (world_point - sensor_pos).Length();
                has_hit = true;
            }
            double cylinder_range = RangeMax();
            if (TerrainCylinderIntersection(sensor_pos, axis_world, cylinder_range) && cylinder_range < hit_range)
            {
                hit_range = cylinder_range;
                world_point = sensor_pos + axis_world * hit_range;
                has_hit = true;
            }
            if (!has_hit)
            {
                continue;
            }

            ignition::math::Vector3d point_sensor =
                sensor_world_rot_inv.RotateVector(world_point - sensor_pos);
            const double range = point_sensor.Length();
            if (range <= RangeMin() || range >= RangeMax())
            {
                continue;
            }
            point_sensor = range * axis_sensor;
            if (IsSelfFiltered(point_sensor))
            {
                continue;
            }

            if (publish_custom)
            {
                livox_ros_driver2::msg::CustomPoint p;
                p.x = point_sensor.X();
                p.y = point_sensor.Y();
                p.z = point_sensor.Z();
                p.reflectivity = 80;
                p.tag = 0x10;
                p.line = static_cast<uint8_t>(count % 4);
                const double phase = unwrapped_pattern_time.size() > 1 ?
                    std::clamp((unwrapped_pattern_time[count] - unwrapped_pattern_time.front()) / pattern_span, 0.0, 1.0) :
                    (static_cast<double>(count) / std::max(1.0, static_cast<double>(total_candidates - 1)));
                p.offset_time = static_cast<uint32_t>(std::clamp(
                    std::llround(phase * static_cast<double>(scan_period_ns)),
                    0LL, static_cast<long long>(std::numeric_limits<uint32_t>::max())));
                pp_livox.points.push_back(p);
            }

            if (publish_cloud2)
            {
                cloud.points.emplace_back();
                cloud.points.back().x = point_sensor.X();
                cloud.points.back().y = point_sensor.Y();
                cloud.points.back().z = point_sensor.Z();
            }
        }

        if (publish_custom)
        {
            pp_livox.point_num = static_cast<uint32_t>(pp_livox.points.size());
            custom_pub->publish(pp_livox);
        }
        if (publish_cloud2)
        {
            sensor_msgs::msg::PointCloud2 cloud2;
            sensor_msgs::convertPointCloudToPointCloud2(cloud, cloud2);
            cloud2.header = cloud.header;
            cloud2_pub->publish(cloud2);
        }
    }

    void LivoxPointsPlugin::InitializeRays(std::vector<std::pair<int, AviaRotateInfo>> &points_pair,
                                           boost::shared_ptr<physics::LivoxOdeMultiRayShape> &ray_shape)
    {
        auto &rays = ray_shape->RayShapes();
        ignition::math::Vector3d start_point, end_point;
        ignition::math::Quaterniond ray;
        auto offset = laserCollision->RelativePose();
        int64_t end_index = currStartIndex + samplesStep;
        long unsigned int ray_index = 0;
        auto ray_size = rays.size();
        points_pair.reserve(rays.size());
        for (int k = currStartIndex; k < end_index; k += downSample)
        {
            auto index = k % maxPointSize;
            auto &rotate_info = aviaInfos[index];
            ray.Euler(ignition::math::Vector3d(0.0, rotate_info.zenith, rotate_info.azimuth));
            auto axis = offset.Rot() * ray * ignition::math::Vector3d(1.0, 0.0, 0.0);
            start_point = minDist * axis + offset.Pos();
            end_point = maxDist * axis + offset.Pos();
            if (ray_index < ray_size)
            {
                rays[ray_index]->SetPoints(start_point, end_point);
                points_pair.emplace_back(ray_index, rotate_info);
            }
            ray_index++;
        }
        currStartIndex += samplesStep;
    }

    void LivoxPointsPlugin::InitializeScan(msgs::LaserScan *&scan)
    {
        // Store the latest laser scans into laserMsg
        msgs::Set(scan->mutable_world_pose(), raySensor->Pose() + parentEntity->WorldPose());
        scan->set_angle_min(AngleMin().Radian());
        scan->set_angle_max(AngleMax().Radian());
        scan->set_angle_step(AngleResolution());
        scan->set_count(RangeCount());

        scan->set_vertical_angle_min(VerticalAngleMin().Radian());
        scan->set_vertical_angle_max(VerticalAngleMax().Radian());
        scan->set_vertical_angle_step(VerticalAngleResolution());
        scan->set_vertical_count(VerticalRangeCount());

        scan->set_range_min(RangeMin());
        scan->set_range_max(RangeMax());

        scan->clear_ranges();
        scan->clear_intensities();

        unsigned int rangeCount = RangeCount();
        unsigned int verticalRangeCount = VerticalRangeCount();

        for (unsigned int j = 0; j < verticalRangeCount; ++j)
        {
            for (unsigned int i = 0; i < rangeCount; ++i)
            {
                scan->add_ranges(0);
                scan->add_intensities(0);
            }
        }
    }

    bool LivoxPointsPlugin::IsSelfFiltered(const ignition::math::Vector3d &point_sensor) const
    {
        if (!self_filter_enable)
            return false;

        const double c = std::cos(self_filter_sensor_yaw);
        const double s = std::sin(self_filter_sensor_yaw);
        const double base_x = self_filter_sensor_x + c * point_sensor.X() - s * point_sensor.Y();
        const double base_y = self_filter_sensor_y + s * point_sensor.X() + c * point_sensor.Y();
        const double base_z = self_filter_sensor_z + point_sensor.Z();

        const double min_x = std::min(self_filter_min_x, self_filter_max_x);
        const double max_x = std::max(self_filter_min_x, self_filter_max_x);
        const double min_y = std::min(self_filter_min_y, self_filter_max_y);
        const double max_y = std::max(self_filter_min_y, self_filter_max_y);
        const double min_z = std::min(self_filter_min_z, self_filter_max_z);
        const double max_z = std::max(self_filter_min_z, self_filter_max_z);

        return base_x >= min_x && base_x <= max_x &&
               base_y >= min_y && base_y <= max_y &&
               base_z >= min_z && base_z <= max_z;
    }

    ignition::math::Angle LivoxPointsPlugin::AngleMin() const
    {
        if (rayShape)
            return rayShape->MinAngle();
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::AngleMax() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->MaxAngle().Radian());
        }
        else
            return -1;
    }

    double LivoxPointsPlugin::GetRangeMin() const { return RangeMin(); }

    double LivoxPointsPlugin::RangeMin() const
    {
        if (rayShape)
            return rayShape->GetMinRange();
        else
            return -1;
    }

    double LivoxPointsPlugin::GetRangeMax() const { return RangeMax(); }

    double LivoxPointsPlugin::RangeMax() const
    {
        if (rayShape)
            return rayShape->GetMaxRange();
        else
            return -1;
    }

    double LivoxPointsPlugin::GetAngleResolution() const { return AngleResolution(); }

    double LivoxPointsPlugin::AngleResolution() const { return (AngleMax() - AngleMin()).Radian() / (RangeCount() - 1); }

    double LivoxPointsPlugin::GetRangeResolution() const { return RangeResolution(); }

    double LivoxPointsPlugin::RangeResolution() const
    {
        if (rayShape)
            return rayShape->GetResRange();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetRayCount() const { return RayCount(); }

    int LivoxPointsPlugin::RayCount() const
    {
        if (rayShape)
            return rayShape->GetSampleCount();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetRangeCount() const { return RangeCount(); }

    int LivoxPointsPlugin::RangeCount() const
    {
        if (rayShape)
            return rayShape->GetSampleCount() * rayShape->GetScanResolution();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetVerticalRayCount() const { return VerticalRayCount(); }

    int LivoxPointsPlugin::VerticalRayCount() const
    {
        if (rayShape)
            return rayShape->GetVerticalSampleCount();
        else
            return -1;
    }

    int LivoxPointsPlugin::GetVerticalRangeCount() const { return VerticalRangeCount(); }

    int LivoxPointsPlugin::VerticalRangeCount() const
    {
        if (rayShape)
            return rayShape->GetVerticalSampleCount() * rayShape->GetVerticalScanResolution();
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::VerticalAngleMin() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->VerticalMinAngle().Radian());
        }
        else
            return -1;
    }

    ignition::math::Angle LivoxPointsPlugin::VerticalAngleMax() const
    {
        if (rayShape)
        {
            return ignition::math::Angle(rayShape->VerticalMaxAngle().Radian());
        }
        else
            return -1;
    }

    double LivoxPointsPlugin::GetVerticalAngleResolution() const { return VerticalAngleResolution(); }

    double LivoxPointsPlugin::VerticalAngleResolution() const
    {
        return (VerticalAngleMax() - VerticalAngleMin()).Radian() / (VerticalRangeCount() - 1);
    }


}
