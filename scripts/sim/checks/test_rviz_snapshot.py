"""Regression checks for 24/32-bit XWD window capture decoding."""

import struct

import numpy as np
import pytest

from rviz_snapshot import read_xwd


@pytest.mark.parametrize("bits_per_pixel", [24, 32])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_decode_xwd_colour_table_stride_and_byte_order(tmp_path, bits_per_pixel, byte_order):
    width, height = 3, 2
    rgb = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[18, 23, 30], [163, 204, 224], [255, 255, 255]],
    ], dtype=np.uint8)
    pixel_bytes = bits_per_pixel // 8
    stride = ((width * pixel_bytes + 3) // 4) * 4
    header = [0] * 25
    header[0], header[1], header[2], header[3] = 108, 7, 2, 24
    header[4], header[5] = width, height
    header[7], header[11], header[12] = byte_order, bits_per_pixel, stride
    header[14:17] = [0xff0000, 0xff00, 0xff]
    header[19] = 2
    rows = bytearray()
    for row in rgb:
        encoded = bytearray()
        for red, green, blue in row:
            value = (int(red) << 16) | (int(green) << 8) | int(blue)
            encoded.extend(value.to_bytes(pixel_bytes, "little" if byte_order == 0 else "big"))
        rows.extend(encoded + bytes(stride - len(encoded)))
    path = tmp_path / "capture.xwd"
    # Both the window name and the colour table precede the actual pixels.
    path.write_bytes(struct.pack(">25I", *header) + b"window\0\0" + bytes(24) + rows)
    assert np.array_equal(np.asarray(read_xwd(path)), rgb)
