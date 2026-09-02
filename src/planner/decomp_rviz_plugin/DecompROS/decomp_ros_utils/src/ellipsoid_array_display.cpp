// ROS2 Humble: Visualization plugin skipped (use rviz2 native plugins instead)
// This file is kept for compatibility only

#include <rclcpp/rclcpp.hpp>

namespace decomp_rviz_plugins {
  // Placeholder for ROS2 compatibility
  class EllipsoidArrayDisplay {
  public:
    EllipsoidArrayDisplay() = default;
    ~EllipsoidArrayDisplay() = default;
  };
}


  if (visual_)
    visual_->setColor(color.r, color.g, color.b, alpha);
}

void EllipsoidArrayDisplay::processMessage(const decomp_ros_msgs::EllipsoidArray::ConstPtr &msg) {
  Ogre::Quaternion orientation;
  Ogre::Vector3 position;
  if (!context_->getFrameManager()->getTransform(
          msg->header.frame_id, msg->header.stamp, position, orientation)) {
    ROS_DEBUG("Error transforming from frame '%s' to frame '%s'",
              msg->header.frame_id.c_str(), qPrintable(fixed_frame_));
    return;
  }

  std::shared_ptr<EllipsoidArrayVisual> visual;
  visual.reset(new EllipsoidArrayVisual(context_->getSceneManager(), scene_node_));

  visual->setMessage(msg);
  visual->setFramePosition(position);
  visual->setFrameOrientation(orientation);

  float alpha = alpha_property_->getFloat();
  Ogre::ColourValue color = color_property_->getOgreColor();
  visual->setColor(color.r, color.g, color.b, alpha);

  visual_ = visual;
}

}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(decomp_rviz_plugins::EllipsoidArrayDisplay, rviz::Display)
