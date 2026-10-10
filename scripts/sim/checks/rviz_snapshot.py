#!/usr/bin/env python3
"""Capture the configured RViz window without touching unrelated desktop windows."""

import argparse
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


WINDOW_RE = re.compile(
    r"^\s*(0x[0-9a-fA-F]+)\s+\"([^\"]*)\".*?\s(\d+)x(\d+)\+(-?\d+)\+(-?\d+)"
)


def find_window(title_part):
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    candidates = []
    for line in tree.splitlines():
        match = WINDOW_RE.match(line)
        if not match or title_part not in match.group(2):
            continue
        candidates.append((int(match.group(3)) * int(match.group(4)), match))
    if not candidates:
        raise RuntimeError(f"window containing {title_part!r} was not found")
    return max(candidates, key=lambda item: item[0])[1], tree


def read_xwd(path):
    raw = path.read_bytes()
    if len(raw) < 100:
        raise RuntimeError("xwd output is too short")
    header = struct.unpack(">25I", raw[:100])
    # XWD fields are not packed as width/height directly after the file
    # header: format/depth occupy slots 2/3, then width/height are 4/5.
    header_size, width, height = header[0], header[4], header[5]
    byte_order, bits_per_pixel, bytes_per_line = header[7], header[11], header[12]
    red_mask, green_mask, blue_mask = header[14:17]
    ncolors = header[19]
    if bits_per_pixel not in (24, 32) or header_size + ncolors * 12 >= len(raw):
        raise RuntimeError(
            f"unsupported xwd format: {bits_per_pixel}bpp header={header_size} colors={ncolors}"
        )
    pixel_offset = header_size + ncolors * 12
    if bits_per_pixel == 24:
        rows = np.frombuffer(
            raw, dtype=np.uint8, offset=pixel_offset,
            count=height * bytes_per_line,
        ).reshape(height, bytes_per_line)[:, :width * 3]
        pixels = rows.reshape(height, width, 3)

        def channel(mask):
            if not mask:
                return np.zeros((height, width), dtype=np.uint8)
            shift = (mask & -mask).bit_length() - 1
            byte_index = shift // 8 if byte_order == 0 else 2 - shift // 8
            return pixels[:, :, byte_index]
    else:
        dtype = np.dtype("<u4" if byte_order == 0 else ">u4")
        stride_words = bytes_per_line // 4
        pixels = np.frombuffer(
            raw, dtype=dtype, offset=pixel_offset,
            count=height * stride_words,
        ).reshape(height, stride_words)[:, :width]

        def channel(mask):
            if not mask:
                return np.zeros((height, width), dtype=np.uint8)
            shift = (mask & -mask).bit_length() - 1
            values = (pixels & mask) >> shift
            maximum = mask >> shift
            return np.asarray(values * 255 // maximum, dtype=np.uint8)

    rgb = np.dstack((channel(red_mask), channel(green_mask), channel(blue_mask)))
    return Image.fromarray(rgb, mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="hunter_terrain_navigation.rviz")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    match, tree = find_window(args.title)
    window_id = match.group(1)
    width, height = int(match.group(3)), int(match.group(4))
    viewport_width = None
    for line in tree.splitlines():
        if window_id not in line and "Ogre" not in line and '"rviz2"' not in line:
            continue
        child = WINDOW_RE.match(line)
        if child and child.group(1) != window_id:
            child_width = int(child.group(3))
            child_height = int(child.group(4))
            if child_width >= 300 and child_height >= 300:
                viewport_width = max(viewport_width or 0, child_width)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".xwd") as capture:
        subprocess.run(["xwd", "-silent", "-id", window_id, "-out", capture.name], check=True)
        image = read_xwd(Path(capture.name))
    image.save(args.output)
    array = np.asarray(image, dtype=np.uint8)
    # A real frame should contain more than one almost-uniform panel colour.
    spread = float(array.reshape(-1, 3).std(axis=0).mean())
    report = {
        "window_id": window_id,
        "window_size": [width, height],
        "viewport_width": viewport_width,
        "rgb_stddev": round(spread, 2),
        "output": str(args.output),
    }
    print(json.dumps(report, ensure_ascii=True), flush=True)
    if width < 800 or height < 500 or spread < 2.0:
        raise RuntimeError(f"captured RViz frame is not useful: {report}")
    if viewport_width is not None and viewport_width < 700:
        raise RuntimeError(f"RViz render viewport is still too narrow: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
