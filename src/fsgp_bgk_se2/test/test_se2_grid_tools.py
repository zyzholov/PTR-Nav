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


@pytest.mark.parametrize('shape',[(1,7),(8,1),(5,7),(8,9),(0,3),(3,0)])
@pytest.mark.parametrize('view',['contiguous','transpose','strided'])
def test_permutation_is_bit_exact_to_roll_reference_and_owns_output(shape,view):
    bits=np.resize(np.array([0x7fc01234,0x80000000,0,0x7f800000,0xff800000,1,0x3f800000],np.uint32),shape)
    matrix=bits.view(np.float32)
    if view=='transpose':matrix=matrix.T
    if view=='strided':matrix=matrix[::-1,::-1]
    for a,b in [(0,0),(2,3),(-7,12),(17,-24)]:
        msg=SimpleNamespace(start_index_row=a,start_index_col=b)
        world=grid_buffer_to_world(matrix,msg)
        expected=np.ascontiguousarray(np.roll(matrix,(-a,-b),axis=(0,1))[::-1,::-1].T)
        assert world.shape==expected.shape and world.tobytes()==expected.tobytes()
        assert world.flags.c_contiguous and not np.shares_memory(world,matrix)
        native=world_to_grid_buffer(matrix,msg)
        expected=np.roll(matrix.T[::-1,::-1],(a,b),axis=(0,1))
        assert native.shape==expected.shape and native.tobytes()==expected.tobytes()
        assert not np.shares_memory(native,matrix)
