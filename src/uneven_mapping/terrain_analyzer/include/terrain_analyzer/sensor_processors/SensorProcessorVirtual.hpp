#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <memory>
#include <string>

namespace terrain_analyzer
{

class SensorProcessorVirtual
{
    public:
        using Ptr = std::unique_ptr<SensorProcessorVirtual>;

        SensorProcessorVirtual() {}

        virtual ~SensorProcessorVirtual() = default;

        bool process(pcl::PointCloud<pcl::PointXYZ> cloud_world,
                    const Eigen::Matrix4d& bodyTrans,
                    const Eigen::Matrix<double, 6, 6>& bodyPoseCovariance,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudWorld, 
                    Eigen::VectorXf& variances)
        {
            pcl::PointCloud<pcl::PointXYZ> temp_cloud;
            pcl::PointCloud<pcl::PointXYZ> cloud_sensor;

            // Remove NaN
            if (!cloud_world.is_dense)
            {
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(cloud_world, temp_cloud, indices);
                temp_cloud.is_dense = true;
                cloud_world.swap(temp_cloud);
            }

            // Voxel Grid Filter
            if (use_voxel_filter)
            {
                pcl::VoxelGrid<pcl::PointXYZ> voxelGridFilter;
                voxelGridFilter.setInputCloud(cloud_world.makeShared());
                voxelGridFilter.setLeafSize(voxel_size, voxel_size, voxel_size);
                voxelGridFilter.filter(temp_cloud);
                cloud_world.swap(temp_cloud);
            }

            // Transform to Body and Filter Z-axis
            pointCloudWorld->clear();
            Eigen::Affine3f bodyTransAf(bodyTrans.cast<float>());
            pcl::transformPointCloud(cloud_world, temp_cloud, bodyTransAf.inverse());
            for (int i=0; i<temp_cloud.size(); i++)
            {
                if (temp_cloud.points[i].z > ignore_z_min && temp_cloud.points[i].z < ignore_z_max)
                    pointCloudWorld->points.emplace_back(temp_cloud.points[i]);
            }
            pointCloudWorld->is_dense = true;
            pointCloudWorld->width = pointCloudWorld->points.size();
            pointCloudWorld->height = 1;
            pointCloudWorld->header.frame_id = "world";
            
            // Transform to Sensor and Map
            pcl::transformPointCloud(*pointCloudWorld, cloud_sensor, body2SensorTrans);
            pcl::transformPointCloud(*pointCloudWorld, *pointCloudWorld, bodyTransAf);

            return computeVariances(cloud_sensor.makeShared(), bodyTrans, bodyPoseCovariance, variances);
        }

        bool init(ros::NodeHandle& nh)
        {
            std::vector<float> sensor2BodyR;
            std::vector<float> sensor2BodyT;
            Eigen::Matrix4f body2SensorRT = Eigen::Matrix4f::Identity();

            nh.getParam("sensor_processors/use_voxel_filter", use_voxel_filter);
            nh.getParam("sensor_processors/voxel_size", voxel_size);
            nh.getParam("sensor_processors/ignore_z_max", ignore_z_max);
            nh.getParam("sensor_processors/ignore_z_min", ignore_z_min);
            nh.param<std::vector<float>>("sensor_processors/sensor2BodyR", sensor2BodyR, std::vector<float>());
            nh.param<std::vector<float>>("sensor_processors/sensor2BodyT", sensor2BodyT, std::vector<float>());
            for (int i=0; i<3; i++)
            {
                for (int j=0; j<3; j++)
                    body2SensorRT(j, i) = sensor2BodyR[i*3+j];
                body2SensorRT(3, i) = -sensor2BodyT[i];
            }
            sensor2BodyRf = body2SensorRT.topLeftCorner(3, 3).transpose();
            sensor2BodyTf = body2SensorRT.topRightCorner(3, 1);
            body2SensorRT.topRightCorner(3, 1) = body2SensorRT.topLeftCorner(3, 3) * body2SensorRT.topRightCorner(3, 1);
            body2SensorTrans = Eigen::Affine3f(body2SensorRT);

            return initSensorParam(nh);
        }

        Eigen::Matrix3f skewSym(Eigen::Vector3f vec)
        {
            Eigen::Matrix3f skem_sym;
            skem_sym << 0.0    , -vec(2), vec(1) , \
                        vec(2) , 0.0    , -vec(0), \
                        -vec(1), vec(0) , 0.0       ;
            return skem_sym;
        }

        virtual bool initSensorParam(ros::NodeHandle& nh) { return true; }

        virtual bool computeVariances(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointCloudSensor, 
                                        const Eigen::Matrix4d& bodyTrans,
                                        const Eigen::Matrix<double, 6, 6>& bodyPoseCovariance,
                                        Eigen::VectorXf& variances) = 0;

    protected:
        Eigen::Affine3f body2SensorTrans;

        Eigen::Matrix3f sensor2BodyRf;

        Eigen::Vector3f sensor2BodyTf;

        bool use_voxel_filter;

        double voxel_size;

        double ignore_z_max;

        double ignore_z_min;
};

} /* namespace terrain_analyzer */
