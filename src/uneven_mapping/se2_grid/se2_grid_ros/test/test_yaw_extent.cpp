#include <gtest/gtest.h>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>

TEST(YawExtent, CanonicalThirtySixBinsRoundTrip)
{
  se2_grid::SE2Grid source;
  source.initGeometry(Eigen::Array2d(4,4),.2,2*M_PI/36,Eigen::Vector2d::Zero(),2*M_PI);
  source.add("risk",true,.25);source.add("valid",false,1);
  se2_grid_msgs::msg::SE2Grid message;
  se2_grid::SE2GridRosConverter::toMessage(source,message);
  ASSERT_EQ(message.data.size(),37u);
  se2_grid::SE2Grid decoded;
  ASSERT_TRUE(se2_grid::SE2GridRosConverter::fromMessage(message,decoded));
  EXPECT_EQ(decoded.getSizeYaw(),36);
  EXPECT_DOUBLE_EQ(decoded.getLengthYaw(),2*M_PI);
  EXPECT_NEAR(decoded.getValue("risk",Eigen::Vector3d(0,0,-.001)),.25,1e-6);
}

TEST(YawExtent,CompactFrontendPreservesEveryRiskSliceAndMeasuredBuffer) {
  se2_grid::SE2Grid source;
  source.initGeometry(Eigen::Array2d(4,4),.2,2*M_PI/36,Eigen::Vector2d::Zero(),2*M_PI);
  source.add("elevation",false,-.4);source.add("ground_observed",false,1.);
  source.add("risk",true,0.);source.add("zbx",true,.12);source.add("zby",true,.13);
  for(int k=0;k<36;++k)source["risk"][k].setConstant(k/36.f);
  source.setStartIndex(Eigen::Array2i(3,7));
  se2_grid_msgs::msg::SE2Grid full,compact;
  se2_grid::SE2GridRosConverter::toMessage(source,full);
  se2_grid::SE2GridRosConverter::toMessage(source,{"elevation","ground_observed","risk"},compact);
  ASSERT_EQ(compact.data.size(),38u);ASSERT_EQ(full.data.size(),110u);
  se2_grid::SE2Grid decoded;
  ASSERT_TRUE(se2_grid::SE2GridRosConverter::fromMessage(compact,decoded));
  EXPECT_EQ(decoded.getSizeYaw(),36);EXPECT_TRUE((decoded.getStartIndex()==source.getStartIndex()).all());
  EXPECT_FALSE(decoded.exists("zbx"));EXPECT_FALSE(decoded.exists("zby"));
  for(const auto &name:compact.layers)for(size_t k=0;k<source[name].size();++k)
    EXPECT_TRUE((source[name][k].array()==decoded[name][k].array()).all())<<name<<k;
  // Output messages own their numeric storage. The zero-copy source read
  // during conversion cannot make later source writes alter sent data.
  source["risk"][0].setConstant(99.);
  EXPECT_FLOAT_EQ(compact.data[2].data[0],0.);
}

TEST(YawExtent, LegacyExtendedCircleIsPreserved)
{
  se2_grid::SE2Grid source;
  source.initGeometry(Eigen::Array2d(4,4),.2,2*M_PI/36);
  source.add("risk",true,.25);source.add("valid",false,1);
  se2_grid_msgs::msg::SE2Grid message;
  se2_grid::SE2GridRosConverter::toMessage(source,message);
  se2_grid::SE2Grid decoded;
  ASSERT_TRUE(se2_grid::SE2GridRosConverter::fromMessage(message,decoded));
  EXPECT_EQ(decoded.getSizeYaw(),37);
  EXPECT_DOUBLE_EQ(decoded.getLengthYaw(),2*M_PI+.02);
}

TEST(YawExtent, MissingAdvertisedSliceIsRejected)
{
  se2_grid::SE2Grid source;
  source.initGeometry(Eigen::Array2d(4,4),.2,2*M_PI/36);
  source.add("risk",true,.25);
  se2_grid_msgs::msg::SE2Grid message;
  se2_grid::SE2GridRosConverter::toMessage(source,message);
  message.data.pop_back();
  se2_grid::SE2Grid decoded;
  EXPECT_FALSE(se2_grid::SE2GridRosConverter::fromMessage(message,decoded));
}

TEST(YawExtent, FullCircleBoundaryIsPeriodic)
{
  se2_grid::SE2Grid grid;
  grid.initGeometry(Eigen::Array2d(4,4),.2,2*M_PI/36,Eigen::Vector2d::Zero(),2*M_PI);
  grid.add("risk",true,.25);
  for (double angle:{2*M_PI,-2*M_PI,6*M_PI,-1e-20}) {
    int index=-1;grid.yaw2Index(angle,index);
    EXPECT_EQ(index,0);
    EXPECT_NEAR(grid.getValue("risk",Eigen::Vector3d(0,0,angle)),.25,1e-6);
  }
}
