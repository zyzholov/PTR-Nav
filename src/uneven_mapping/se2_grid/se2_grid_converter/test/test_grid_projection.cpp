#include "se2_grid_converter/grid_projection.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

namespace
{

std_msgs::msg::Float32MultiArray makeRowMajor(std::size_t rows, std::size_t cols, std::size_t offset)
{
  std_msgs::msg::Float32MultiArray msg;
  msg.layout.data_offset = offset;
  msg.layout.dim.resize(2);
  msg.layout.dim[0].label = "row_index";
  msg.layout.dim[0].size = rows;
  msg.layout.dim[0].stride = rows * cols;
  msg.layout.dim[1].label = "column_index";
  msg.layout.dim[1].size = cols;
  msg.layout.dim[1].stride = cols;
  msg.data.resize(offset + rows * cols, -1.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      msg.data[offset + row * cols + col] = static_cast<float>(10 * row + col);
    }
  }
  return msg;
}

std_msgs::msg::Float32MultiArray makeColumnMajor(std::size_t rows, std::size_t cols)
{
  std_msgs::msg::Float32MultiArray msg;
  msg.layout.dim.resize(2);
  msg.layout.dim[0].label = "column_index";
  msg.layout.dim[0].size = cols;
  msg.layout.dim[0].stride = rows * cols;
  msg.layout.dim[1].label = "row_index";
  msg.layout.dim[1].size = rows;
  msg.layout.dim[1].stride = rows;
  msg.data.resize(rows * cols, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      msg.data[col * rows + row] = static_cast<float>(10 * row + col);
    }
  }
  return msg;
}

}  // namespace

TEST(GridProjection, ReadsRowMajorWithOffsetAndUnwrapsCircularIndex)
{
  const auto msg = makeRowMajor(3, 5, 2);
  se2_grid_converter::MatrixLayout layout;
  std::string error;
  ASSERT_TRUE(se2_grid_converter::decodeLayout(msg, 3, 5, layout, error)) << error;

  float value = 0.0F;
  ASSERT_TRUE(se2_grid_converter::readValue(msg, layout, 2, 4, value));
  EXPECT_FLOAT_EQ(value, 24.0F);
  EXPECT_DOUBLE_EQ(
    se2_grid_converter::cellCoordinate(10.0, 1.5, 0.5, 0, 1, 3), 9.5);
  EXPECT_DOUBLE_EQ(
    se2_grid_converter::cellCoordinate(20.0, 2.5, 0.5, 3, 3, 5), 21.0);
}

TEST(GridProjection, ReadsColumnMajorUsingNativeRowsAndColumns)
{
  const auto msg = makeColumnMajor(3, 5);
  se2_grid_converter::MatrixLayout layout;
  std::string error;
  ASSERT_TRUE(se2_grid_converter::decodeLayout(msg, 3, 5, layout, error)) << error;

  float value = 0.0F;
  ASSERT_TRUE(se2_grid_converter::readValue(msg, layout, 2, 4, value));
  EXPECT_FLOAT_EQ(value, 24.0F);
}

TEST(GridProjection, RejectsWrongGeometryInsteadOfSilentlyTransposing)
{
  const auto msg = makeRowMajor(3, 5, 0);
  se2_grid_converter::MatrixLayout layout;
  std::string error;
  EXPECT_FALSE(se2_grid_converter::decodeLayout(msg, 5, 3, layout, error));
  EXPECT_FALSE(error.empty());
}

TEST(GridProjection, SelectsFloorYawBinAndWrapsNegativeAngles)
{
  constexpr double resolution = 0.5;
  EXPECT_EQ(se2_grid_converter::yawIndex(0.99, resolution, 13), 1);
  EXPECT_EQ(se2_grid_converter::yawIndex(-0.01, resolution, 13), 12);
}
