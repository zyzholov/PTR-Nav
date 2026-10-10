from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


ROW_LABEL = "row_index"
COL_LABEL = "column_index"


@dataclass(frozen=True)
class LayerRef:
    name: str
    has_so2: bool
    data_indices: Tuple[int, ...]


def yaw_slices(msg) -> int:
    yaw_res = max(float(msg.info.yaw_resolution), 1.0e-6)
    length_yaw = max(float(msg.info.length_yaw), yaw_res)
    return max(1, int(math.ceil(length_yaw / yaw_res - 1.0e-6)))


def layer_refs(msg) -> Dict[str, LayerRef]:
    refs: Dict[str, LayerRef] = {}
    yaws = yaw_slices(msg)
    data_idx = 0
    for i, name in enumerate(msg.layers):
        so2 = bool(msg.has_so2[i]) if i < len(msg.has_so2) else False
        count = yaws if so2 else 1
        refs[name] = LayerRef(
            name=name,
            has_so2=so2,
            data_indices=tuple(range(data_idx, data_idx + count)),
        )
        data_idx += count
    return refs


def _is_row_major(array_msg: Float32MultiArray) -> bool:
    if len(array_msg.layout.dim) < 2:
        return True
    return array_msg.layout.dim[0].label == ROW_LABEL


def matrix_shape(array_msg: Float32MultiArray) -> Tuple[int, int]:
    if len(array_msg.layout.dim) >= 2:
        first = array_msg.layout.dim[0]
        second = array_msg.layout.dim[1]
        if first.label == ROW_LABEL:
            return int(first.size), int(second.size)
        return int(second.size), int(first.size)

    side = int(round(np.sqrt(len(array_msg.data))))
    if side * side != len(array_msg.data):
        raise ValueError("Cannot infer matrix shape from a flat Float32MultiArray")
    return side, side


def multiarray_to_matrix(array_msg: Float32MultiArray) -> np.ndarray:
    rows, cols = matrix_shape(array_msg)
    data = np.asarray(array_msg.data, dtype=np.float32)
    order = "C" if _is_row_major(array_msg) else "F"
    if data.size != rows * cols:
        raise ValueError(
            f"SE2Grid layer data size mismatch: got {data.size}, expected {rows * cols}"
        )
    return np.reshape(data, (rows, cols), order=order).copy()


def matrix_to_multiarray_like(
    matrix: np.ndarray,
    template: Optional[Float32MultiArray],
) -> Float32MultiArray:
    rows, cols = matrix.shape
    out = Float32MultiArray()
    if template is not None and len(template.layout.dim) >= 2:
        out.layout = template.layout
        # The output buffer below is packed; do not retain padding from an
        # input message that happened to use a non-zero data offset.
        out.layout.data_offset = 0
        order = "C" if _is_row_major(template) else "F"
    else:
        out.layout.dim = [
            MultiArrayDimension(label=COL_LABEL, size=cols, stride=rows * cols),
            MultiArrayDimension(label=ROW_LABEL, size=rows, stride=rows),
        ]
        out.layout.data_offset = 0
        order = "F"

    values = array("f")
    values.frombytes(np.asarray(matrix, dtype=np.float32).tobytes(order=order))
    out.data = values
    return out


def first_data_template(msg) -> Optional[Float32MultiArray]:
    return msg.data[0] if msg.data else None


def grid_buffer_to_world(matrix: np.ndarray, msg) -> np.ndarray:
    # Native SE2Grid rows/columns are descending X/Y in a circular buffer.
    unwrapped = np.roll(
        matrix, (-int(msg.start_index_row), -int(msg.start_index_col)), axis=(0, 1)
    )
    return np.ascontiguousarray(unwrapped[::-1, ::-1].T)


def world_to_grid_buffer(matrix: np.ndarray, msg) -> np.ndarray:
    native = matrix.T[::-1, ::-1]
    return np.roll(
        native, (int(msg.start_index_row), int(msg.start_index_col)), axis=(0, 1)
    )


def get_layer_slices(msg, name: str) -> Tuple[List[np.ndarray], bool, Optional[Float32MultiArray]]:
    refs = layer_refs(msg)
    ref = refs.get(name)
    if ref is None:
        return [], False, first_data_template(msg)

    matrices: List[np.ndarray] = []
    template: Optional[Float32MultiArray] = None
    for idx in ref.data_indices:
        if idx >= len(msg.data):
            break
        template = msg.data[idx]
        matrices.append(multiarray_to_matrix(msg.data[idx]))
    return matrices, ref.has_so2, template


def get_layer_2d(
    msg,
    name: str,
    default_value: float,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Optional[Float32MultiArray]]:
    slices, _, template = get_layer_slices(msg, name)
    if slices:
        return slices[0], template

    if shape is None:
        template = first_data_template(msg)
        if template is None:
            raise ValueError(f"Layer '{name}' is missing and no shape is available")
        shape = matrix_shape(template)
    return np.full(shape, default_value, dtype=np.float32), template


def repeat_or_trim_slices(
    slices: Sequence[np.ndarray],
    count: int,
    fallback: np.ndarray,
) -> List[np.ndarray]:
    if not slices:
        return [fallback.copy() for _ in range(count)]
    if len(slices) == 1:
        return [slices[0].copy() for _ in range(count)]
    out = [s.copy() for s in slices[:count]]
    while len(out) < count:
        out.append(out[-1].copy())
    return out
