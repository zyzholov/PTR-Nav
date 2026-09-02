#pragma once

#ifndef Q_MOC_RUN
#include <se2_grid_msgs/SE2Grid.h>
#include <boost/circular_buffer.hpp>
// The following replaces <rviz/message_filter_display.h>
#include "se2_grid_rviz_plugin/modified/message_filter_display.h"
#endif

namespace Ogre
{
    class SceneNode;
}

namespace rviz
{
    class BoolProperty;
    class ColorProperty;
    class FloatProperty;
    class IntProperty;
    class EnumProperty;
    class EditableEnumProperty;
}

namespace se2_grid_rviz_plugin {

class SE2GridVisual;
class SE2GridDisplay : public MessageFilterDisplay<se2_grid_msgs::SE2Grid>
{
Q_OBJECT
 public:
  SE2GridDisplay();
  virtual ~SE2GridDisplay();

 protected:
  virtual void onInitialize();

  virtual void onEnable();

  virtual void reset();

 Q_SIGNALS:
  // Signal to ensure that the rendering happens in the ui thread.
  void process(const se2_grid_msgs::SE2Grid::ConstPtr& msg);

 private Q_SLOTS:
  void updateHistoryLength();
  void updateHeightMode();
  void updateColorMode();
  void updateUseColorMap();
  void updateAutocomputeIntensityBounds();
  void updateVisualization();
  void updateColorMapList();
  void updateGridLines();
  // Slot to ensure that the rendering happens in the ui thread.
  void onProcessMessage(const se2_grid_msgs::SE2Grid::ConstPtr& msg);

 private:
  // Callback for incoming ROS messages
  void processMessage(const se2_grid_msgs::SE2Grid::ConstPtr& msg);

  // Flag to ensure that after the reset the scene is not updated again.
  std::atomic<bool> isReset_{false};

  // Circular buffer for visuals
  boost::circular_buffer<boost::shared_ptr<SE2GridVisual> > visuals_;

  // Property variables
  rviz::FloatProperty* alphaProperty_;
  rviz::IntProperty* historyLengthProperty_;
  rviz::FloatProperty* so2ValueProperty_;
  rviz::BoolProperty* showGridLinesProperty_;
  rviz::EnumProperty* heightModeProperty_;
  rviz::EditableEnumProperty* heightTransformerProperty_;
  rviz::EnumProperty* colorModeProperty_;
  rviz::EditableEnumProperty* colorTransformerProperty_;
  rviz::EditableEnumProperty* colorMapProperty_;
  rviz::ColorProperty* colorProperty_;
  rviz::BoolProperty* useColorMapProperty_;
  rviz::BoolProperty* invertColorMapProperty_;
  rviz::ColorProperty* minColorProperty_;
  rviz::ColorProperty* maxColorProperty_;
  rviz::BoolProperty* autocomputeIntensityBoundsProperty_;
  rviz::FloatProperty* minIntensityProperty_;
  rviz::FloatProperty* maxIntensityProperty_;
  rviz::FloatProperty* gridLinesThicknessProperty_;
};

}  // end namespace se2_grid_rviz_plugin
