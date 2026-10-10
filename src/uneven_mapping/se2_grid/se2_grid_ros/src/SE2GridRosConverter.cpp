#include "se2_grid_ros/SE2GridRosConverter.hpp"
#include "se2_grid_ros/SE2GridMsgHelpers.hpp"

// ROS Message types
#include <se2_grid_msgs/msg/detail/se2_grid__struct.hpp>

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// STL
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;
using namespace Eigen;

namespace se2_grid {

SE2GridRosConverter::SE2GridRosConverter()
{
}

SE2GridRosConverter::~SE2GridRosConverter()
{
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& /*layers*/)
{
  if (message.layers.size() != message.has_so2.size())
  {
    cerr << "SE2Grid message layer/has_so2 size mismatch: layers=" << message.layers.size()
         << " has_so2=" << message.has_so2.size() << endl;
    return false;
  }

  SE2Grid.initGeometry(Eigen::Array2d(message.info.length_x, message.info.length_y), 
                       message.info.pos_resolution, message.info.yaw_resolution,
                       Eigen::Vector2d(message.info.pose.position.x, message.info.pose.position.y));

  // Copy non-basic layers.
  unsigned int data_idx = 0;
  for (unsigned int i = 0u; i < message.layers.size(); ++i)
  {
    // TODO Could we use the data mapping (instead of copying) method here?
    std::vector<Eigen::MatrixXf> datas;
    int size_yaw = SE2Grid.getSizeYaw();
    if (message.has_so2[i])
    {
      if (data_idx + static_cast<unsigned int>(size_yaw) > message.data.size())
      {
        cerr << "SE2Grid message missing SO2 data for layer '" << message.layers[i]
             << "': need " << size_yaw << " matrices at data index " << data_idx
             << ", total data=" << message.data.size() << endl;
        return false;
      }
      for (int j=0; j<size_yaw; j++)
      {
        Eigen::MatrixXf d;
        if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
        {
          return false;
        }
        datas.emplace_back(d);
      }
      SE2Grid.add(message.layers[i], datas);
    }
    else
    {
      if (data_idx >= message.data.size())
      {
        cerr << "SE2Grid message missing 2D data for layer '" << message.layers[i]
             << "' at data index " << data_idx
             << ", total data=" << message.data.size() << endl;
        return false;
      }
      Eigen::MatrixXf d;
      if(!multiArrayMessageCopyToMatrixEigen(message.data[data_idx++], d))
      {
        return false;
      }
      datas.emplace_back(d);
      SE2Grid.add(message.layers[i], datas);
    }
  }

  SE2Grid.setStartIndex(Eigen::Array2i(message.start_index_row, message.start_index_col));

  return true;
}

bool SE2GridRosConverter::fromMessage(const se2_grid_msgs::msg::SE2Grid& message, se2_grid::SE2Grid& SE2Grid)
{
  return fromMessage(message, SE2Grid, std::vector<std::string>());
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, se2_grid_msgs::msg::SE2Grid& message)
{
  toMessage(SE2Grid, SE2Grid.getLayers(), message);
}

void SE2GridRosConverter::toMessage(const se2_grid::SE2Grid& SE2Grid, const std::vector<std::string>& layers,
                                    se2_grid_msgs::msg::SE2Grid& message)
{
  message.info.header.stamp = rclcpp::Clock(RCL_SYSTEM_TIME).now(); // TODO: check if it's correct clock
  message.info.header.frame_id = SE2Grid.getFrameId();
  message.info.pos_resolution = SE2Grid.getResolutionPos();
  message.info.yaw_resolution = SE2Grid.getResolutionYaw();
  message.info.length_x = SE2Grid.getLengthPos().x();
  message.info.length_y = SE2Grid.getLengthPos().y();
  message.info.length_yaw = SE2Grid.getLengthYaw();
  message.info.pose.position.x = SE2Grid.getPosition().x();
  message.info.pose.position.y = SE2Grid.getPosition().y();
  message.info.pose.position.z = 0.0;
  message.info.pose.orientation.x = 0.0;
  message.info.pose.orientation.y = 0.0;
  message.info.pose.orientation.z = 0.0;
  message.info.pose.orientation.w = 1.0;

  message.layers = layers;

  message.data.clear();
  message.has_so2.clear();
  for (size_t i=0; i<layers.size(); i++)
  {
    std_msgs::msg::Float32MultiArray dataArray;
    if (SE2Grid.hasSO2(layers[i]))
      message.has_so2.emplace_back(true);
    else
      message.has_so2.emplace_back(false);

    std::vector<Eigen::MatrixXf> l = SE2Grid[layers[i]];
    for (size_t j=0; j<l.size(); j++)
    {
      matrixEigenCopyToMultiArrayMessage(l[j], dataArray);
      message.data.push_back(dataArray);
    }
  }

  message.start_index_row = SE2Grid.getStartIndex()(0);
  message.start_index_col = SE2Grid.getStartIndex()(1);
}// ROS1 rosbag functions disabled for ROS2 compatibility
bool SE2GridRosConverter::saveToBag(const se2_grid::SE2Grid& SE2Grid, const std::string& pathToBag,
                                    const std::string& topic)
{
  // ROS2 does not support rosbag in the same way as ROS1
  RCLCPP_WARN(rclcpp::get_logger("se2_grid"), "saveToBag is not supported in ROS2. Returning false.");
  return false;
}

bool SE2GridRosConverter::loadFromBag(const std::string& pathToBag, const std::string& topic,
                                      se2_grid::SE2Grid& SE2Grid)
{
  // ROS2 does not support rosbag in the same way as ROS1
  RCLCPP_WARN(rclcpp::get_logger("se2_grid"), "loadFromBag is not supported in ROS2. Returning false.");
  return false;
}

} /* namespace */
