// ROS2 Humble: Visualization not supported - stub implementation
// This file is kept for build compatibility only
#include <rclcpp/rclcpp.hpp>

    frame_node_->setOrientation(orientation);
  }

  void EllipsoidArrayVisual::setColor(float r, float g, float b, float a) {
    for (auto &it : objs_)
      it->setColor(r, g, b, a);
  }
}
