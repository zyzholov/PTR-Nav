from __future__ import annotations

from array import array
from dataclasses import dataclass
from functools import lru_cache
import math
import struct
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from types import MappingProxyType

import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header


ROW_LABEL = "row_index"
COL_LABEL = "column_index"


def serialized_header(payload):
    """Leading Header for Humble CDR1 types, including SE2Grid.info.header.

    Check the envelope before selecting a large message for deserialization.
    Unsupported encodings fail rather than guessing a time stamp.
    """
    if len(payload)<16 or bytes(payload[:2]) not in (b'\x00\x00',b'\x00\x01'):
        raise ValueError('expected a CDR1 leading Header')
    order='<' if payload[1]==1 else '>'
    sec,nanosec,frame_size=struct.unpack_from(order+'iII',payload,4)
    if nanosec>=1_000_000_000 or frame_size<1 or 16+frame_size>len(payload) or payload[15+frame_size]!=0:
        raise ValueError('invalid serialized Header')
    header=Header();header.stamp.sec=sec;header.stamp.nanosec=nanosec
    header.frame_id=bytes(payload[16:15+frame_size]).decode('utf-8')
    return header


@dataclass(frozen=True)
class LayerRef:
    name: str
    has_so2: bool
    data_indices: Tuple[int, ...]


def yaw_slices(msg) -> int:
    yaw_res = max(float(msg.info.yaw_resolution), 1.0e-6)
    length_yaw = max(float(msg.info.length_yaw), yaw_res)
    return max(1, int(math.ceil(length_yaw / yaw_res - 1.0e-6)))


def layer_refs(msg) -> Mapping[str, LayerRef]:
    # Cache only the immutable SCHEMA, never a message, stamp, buffer, pose or
    # risk value. Append/change/reorder/yaw-schema changes get a different key.
    return _schema_refs(tuple(msg.layers),tuple(bool(v) for v in msg.has_so2),yaw_slices(msg))


@lru_cache(maxsize=32)
def _schema_refs(layers,has_so2,yaws):
    refs: Dict[str, LayerRef] = {}
    data_idx = 0
    for i, name in enumerate(layers):
        so2 = has_so2[i] if i < len(has_so2) else False
        count = yaws if so2 else 1
        refs[name] = LayerRef(
            name=name,
            has_so2=so2,
            data_indices=tuple(range(data_idx, data_idx + count)),
        )
        data_idx += count
    return MappingProxyType(refs)


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


@lru_cache(maxsize=1)
def _packed_array_schema_supported():
    # Exact generated Humble schema only. Future/generated variants retain
    # their public constructors/setters instead of receiving partial objects.
    return (tuple(Float32MultiArray.__slots__)==('_layout','_data') and
        tuple(MultiArrayLayout.__slots__)==('_dim','_data_offset') and
        Float32MultiArray.get_fields_and_field_types()==dict(layout='std_msgs/MultiArrayLayout',data='sequence<float>') and
        MultiArrayLayout.get_fields_and_field_types()==dict(dim='sequence<std_msgs/MultiArrayDimension>',data_offset='uint32'))


def matrix_to_multiarray_like(
    matrix: np.ndarray,
    template: Optional[Float32MultiArray],
) -> Float32MultiArray:
    rows, cols = matrix.shape
    fast=_packed_array_schema_supported()
    if fast:
        # The schema is known and every slot is set below. Avoid creating
        # empty default lists/arrays and validating the same received layout
        # hundreds of times per map. Layout and numeric data still have
        # private ownership; no wire field or float bit is omitted.
        out=Float32MultiArray.__new__(Float32MultiArray)
        layout=MultiArrayLayout.__new__(MultiArrayLayout)
        layout._data_offset=0
        out._layout=layout
    else:out = Float32MultiArray()
    if template is not None and len(template.layout.dim) >= 2:
        # The template may still belong to a received map shared by another
        # callback. Own the layout before resetting the packed data offset;
        # immutable dimensions can be shared without copying numeric buffers.
        # Float32MultiArray already constructed a private layout. Reuse it
        # instead of constructing and immediately discarding a second one
        # for each of hundreds of matrices in a map.
        if fast and isinstance(template,Float32MultiArray):out.layout._dim=template.layout.dim
        else:out.layout.dim = template.layout.dim
        # The output buffer below is packed; do not retain padding from an
        # input message that happened to use a non-zero data offset.
        if not fast:out.layout.data_offset = 0
        order = "C" if _is_row_major(template) else "F"
    else:
        dimensions = [
            MultiArrayDimension(label=COL_LABEL, size=cols, stride=rows * cols),
            MultiArrayDimension(label=ROW_LABEL, size=rows, stride=rows),
        ]
        if fast:out.layout._dim=dimensions
        else:
            out.layout.dim=dimensions
            out.layout.data_offset = 0
        order = "F"

    values = array("f")
    # A one-dimensional packed ndarray exposes the exact native Float32
    # buffer in the requested storage order. frombytes makes the output's
    # one owning copy; an intermediate Python bytes copy is unnecessary.
    packed=np.asarray(matrix,dtype=np.float32).ravel(order=order)
    values.frombytes(memoryview(packed).cast('B'))
    if fast:out._data=values
    else:out.data = values
    return out


def first_data_template(msg) -> Optional[Float32MultiArray]:
    return msg.data[0] if msg.data else None


def _roll2d_owned(matrix, row_shift, col_shift):
    """Same owned 2-D permutation as np.roll, without its generic axis setup."""
    rows,cols=matrix.shape
    if not rows or not cols:return matrix.copy(order='K')
    r,c=int(row_shift)%rows,int(col_shift)%cols
    out=np.empty_like(matrix)
    out[r:,c:]=matrix[:rows-r,:cols-c]
    if r:out[:r,c:]=matrix[rows-r:,:cols-c]
    if c:out[r:,:c]=matrix[:rows-r,cols-c:]
    if r and c:out[:r,:c]=matrix[rows-r:,cols-c:]
    return out


def grid_buffer_to_world(matrix: np.ndarray, msg) -> np.ndarray:
    # Native SE2Grid rows/columns are descending X/Y in a circular buffer.
    unwrapped = _roll2d_owned(matrix,-int(msg.start_index_row),-int(msg.start_index_col))
    return np.ascontiguousarray(unwrapped[::-1, ::-1].T)


def world_to_grid_buffer(matrix: np.ndarray, msg) -> np.ndarray:
    # Retain native F-contiguous layout: the wire is normally column-major.
    # A C-contiguous advanced-index destination makes later packing copy it
    # again and degraded measured end-to-end stage timings.
    native = matrix.T[::-1, ::-1]
    return _roll2d_owned(native,int(msg.start_index_row),int(msg.start_index_col))


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
    # Match circular angular bin centers, not a prefix padded with the last yaw.
    return [slices[min(len(slices)-1, int((i+.5)*len(slices)/count))].copy()
            for i in range(count)]
