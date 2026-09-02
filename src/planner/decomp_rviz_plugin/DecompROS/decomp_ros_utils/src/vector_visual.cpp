// ROS2 Humble: Visualization not supported in this compatibility build
#include <rclcpp/rclcpp.hpp>

namespace decomp_rviz_plugins {
  // Placeholder implementation
  class VectorVisual {
  public:
    VectorVisual() = default;
    ~VectorVisual() = default;
  };
}

    int cnt = 0;
    for (const auto &v : vs) {
      Vec3f n = v.second.normalized();
      objs_[cnt]->setDirection(Ogre::Vector3(n(0), n(1), n(2)));
      objs_[cnt]->setPosition(Ogre::Vector3(v.first(0), v.first(1), v.first(2)));
      objs_[cnt]->setScale(s_ * Ogre::Vector3(1.0, 1.0, 1.0));

      float d = s_ * v.second.norm();
      float shaft_length = 0.7 * d;
      float head_length = 0.3 * d;
      objs_[cnt]->set(shaft_length, d / 8, head_length, d / 5);
      cnt ++;
    }
  }

  void VectorVisual::setFramePosition(const Ogre::Vector3 &position) {
    frame_node_->setPosition(position);
  }

  void VectorVisual::setFrameOrientation(const Ogre::Quaternion &orientation) {
    frame_node_->setOrientation(orientation);
  }

  void VectorVisual::setColor(float r, float g, float b, float a) {
    for (auto &it : objs_)
      it->setColor(r, g, b, a);
  }

  void VectorVisual::setScale(float s) {
    s_ = s;
    for (unsigned int i = 0; i < objs_.size(); i++){
      float d = s_ * vs_[i].second.norm();
      float shaft_length = 0.7 * d;
      float head_length = 0.3 * d;
      objs_[i]->set(shaft_length, d / 8, head_length, d / 5);
    }
  }
}
