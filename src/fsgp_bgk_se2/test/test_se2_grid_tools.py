from array import array
from types import SimpleNamespace

import numpy as np
import pytest
from rclpy.serialization import deserialize_message, serialize_message
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from fsgp_bgk_se2.se2_grid_tools import (
    grid_buffer_to_world,
    matrix_to_multiarray_like,
    multiarray_to_matrix,
    world_to_grid_buffer,
)


@pytest.mark.parametrize("row_major", [False, True])
def test_matrix_multiarray_binary_roundtrip(row_major):
    matrix = np.arange(35, dtype=np.float32).reshape(5, 7) / 7.0
    template = Float32MultiArray()
    if row_major:
        template.layout.dim = [
            MultiArrayDimension(label="row_index", size=5, stride=35),
            MultiArrayDimension(label="column_index", size=7, stride=7),
        ]
    else:
        template.layout.dim = [
            MultiArrayDimension(label="column_index", size=7, stride=35),
            MultiArrayDimension(label="row_index", size=5, stride=5),
        ]

    message = matrix_to_multiarray_like(matrix[:, ::-1], template)
    assert isinstance(message.data, array)
    decoded = deserialize_message(serialize_message(message), Float32MultiArray)
    np.testing.assert_array_equal(multiarray_to_matrix(decoded), matrix[:, ::-1])


def test_matrix_multiarray_without_template():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    message = matrix_to_multiarray_like(matrix, None)
    np.testing.assert_array_equal(multiarray_to_matrix(message), matrix)


@pytest.mark.parametrize("start", [(0, 0), (3, 2), (6, 4)])
def test_world_axes_and_circular_buffer(start):
    msg = SimpleNamespace(start_index_row=start[0], start_index_col=start[1])
    # Seven X cells and five Y cells; native index zero is the maximum corner.
    world = np.arange(7, dtype=np.float32)[None, :] + 100 * np.arange(5)[:, None]
    native = world_to_grid_buffer(world, msg)
    assert native.shape == (7, 5)
    assert native[start[0], start[1]] == world[-1, -1]
    np.testing.assert_array_equal(grid_buffer_to_world(native, msg), world)
