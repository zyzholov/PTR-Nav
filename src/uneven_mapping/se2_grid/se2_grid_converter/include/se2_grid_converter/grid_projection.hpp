#pragma once

#include <std_msgs/msg/float32_multi_array.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace se2_grid_converter
{

struct MatrixLayout
{
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t offset{0};
  std::size_t row_stride{0};
  std::size_t col_stride{0};
  bool row_major{true};
};

inline bool decodeLayout(
  const std_msgs::msg::Float32MultiArray & array,
  std::size_t expected_rows,
  std::size_t expected_cols,
  MatrixLayout & out,
  std::string & error)
{
  if (array.layout.dim.size() < 2) {
    error = "SE2Grid layer has fewer than two layout dimensions";
    return false;
  }

  const auto & first = array.layout.dim[0];
  const auto & second = array.layout.dim[1];
  const bool first_is_row = first.label == "row_index";
  const bool first_is_col = first.label == "column_index";
  if (!first_is_row && !first_is_col) {
    error = "SE2Grid layout dim[0] must be row_index or column_index";
    return false;
  }
  if (first.stride == 0 || second.stride == 0) {
    error = "SE2Grid layer has zero layout stride";
    return false;
  }

  out.row_major = first_is_row;
  out.rows = first_is_row ? first.size : second.size;
  out.cols = first_is_row ? second.size : first.size;
  // The legacy SE2Grid publisher fills MultiArrayDimension::stride with
  // Eigen's outer stride, while ROS MultiArray consumers commonly interpret
  // it as a byte/element jump.  The actual payload is packed in the order
  // described by the labels, so derive the safe element strides from the
  // dimensions and validate the metadata only for presence above.
  out.row_stride = first_is_row ? out.cols : 1;
  out.col_stride = first_is_row ? 1 : out.rows;
  out.offset = array.layout.data_offset;

  if (out.rows != expected_rows || out.cols != expected_cols) {
    error = "SE2Grid layer dimensions do not match grid geometry";
    return false;
  }

  if (out.rows == 0 || out.cols == 0) {
    error = "SE2Grid layer has an empty dimension";
    return false;
  }
  const std::size_t max_index = out.offset +
    (out.rows - 1) * out.row_stride + (out.cols - 1) * out.col_stride;
  if (max_index >= array.data.size()) {
    error = "SE2Grid layer layout points outside its data buffer";
    return false;
  }
  return true;
}

inline bool readValue(
  const std_msgs::msg::Float32MultiArray & array,
  const MatrixLayout & layout,
  std::size_t row,
  std::size_t col,
  float & value)
{
  if (row >= layout.rows || col >= layout.cols) {
    return false;
  }
  const std::size_t index = layout.offset + row * layout.row_stride + col * layout.col_stride;
  if (index >= array.data.size()) {
    return false;
  }
  value = array.data[index];
  return true;
}

inline std::size_t wrappedIndex(std::size_t index, std::size_t start, std::size_t size)
{
  if (size == 0) {
    return 0;
  }
  const auto value = static_cast<std::int64_t>(index) - static_cast<std::int64_t>(start);
  const auto modulus = static_cast<std::int64_t>(size);
  const auto wrapped = (value % modulus + modulus) % modulus;
  return static_cast<std::size_t>(wrapped);
}

inline double cellCoordinate(
  double center,
  double length,
  double resolution,
  std::size_t buffer_index,
  std::size_t start_index,
  std::size_t size)
{
  // SE2Grid stores X/Y indices from the positive edge toward the negative edge.
  const std::size_t true_index = wrappedIndex(buffer_index, start_index, size);
  return center + 0.5 * length - 0.5 * resolution -
    static_cast<double>(true_index) * resolution;
}

inline int yawIndex(double yaw, double yaw_resolution, int yaw_count)
{
  if (yaw_count <= 0 || yaw_resolution <= 0.0 || !std::isfinite(yaw)) {
    return 0;
  }
  constexpr double two_pi = 2.0 * 3.14159265358979323846;
  double wrapped = std::fmod(yaw, two_pi);
  if (wrapped < 0.0) {
    wrapped += two_pi;
  }
  auto index = static_cast<int>(std::floor(wrapped / yaw_resolution));
  return ((index % yaw_count) + yaw_count) % yaw_count;
}

}  // namespace se2_grid_converter
