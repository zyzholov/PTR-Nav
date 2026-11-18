#!/usr/bin/env python3
"""Briefly activate the configured RViz window and capture its visible contents."""

import argparse
from contextlib import contextmanager
import ctypes
import ctypes.util
import json
import re
import struct
import subprocess
import tempfile
import time
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


def render_viewport_width(tree):
    """Only inspect the selected window's subtree, never another RViz instance."""
    widths = []
    for line in tree.splitlines():
        child = WINDOW_RE.match(line)
        if child and ("Ogre" in child.group(2) or child.group(2) == "rviz2"):
            width, height = int(child.group(3)), int(child.group(4))
            if width >= 300 and height >= 300:
                widths.append(width)
    return max(widths, default=None)


class _ClientMessage(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong),
                ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
                ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
                ("format", ctypes.c_int), ("data", ctypes.c_long * 5)]


class _XEvent(ctypes.Union):
    _fields_ = [("client", _ClientMessage), ("pad", ctypes.c_long * 24)]


class XDesktop:
    """Use the window manager activation protocol; no extra desktop packages."""

    def __init__(self):
        self.x = ctypes.CDLL(ctypes.util.find_library("X11"))
        signatures = {
            "XOpenDisplay": ([ctypes.c_char_p], ctypes.c_void_p),
            "XDefaultRootWindow": ([ctypes.c_void_p], ctypes.c_ulong),
            "XInternAtom": ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_ulong),
            "XGetWindowProperty": ([ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            "XSendEvent": ([ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                ctypes.c_long, ctypes.POINTER(_XEvent)], ctypes.c_int),
            "XFree": ([ctypes.c_void_p], ctypes.c_int),
            "XFlush": ([ctypes.c_void_p], ctypes.c_int),
            "XCloseDisplay": ([ctypes.c_void_p], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.x, name)
            function.argtypes, function.restype = arguments, result
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError("cannot connect to X11 display")
        self.root = self.x.XDefaultRootWindow(self.display)
        self.active_atom = self.x.XInternAtom(self.display, b"_NET_ACTIVE_WINDOW", False)

    def active(self):
        actual_type, count, remaining = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_ulong()
        actual_format, data = ctypes.c_int(), ctypes.c_void_p()
        status = self.x.XGetWindowProperty(self.display, self.root, self.active_atom,
            0, 1, False, 33, ctypes.byref(actual_type), ctypes.byref(actual_format),
            ctypes.byref(count), ctypes.byref(remaining), ctypes.byref(data))
        try:
            if status or actual_type.value != 33 or actual_format.value != 32 or not count.value or not data.value:
                return 0
            return ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))[0]
        finally:
            if data.value:
                self.x.XFree(data)

    def activate(self, window):
        event = _XEvent()
        event.client.type, event.client.send_event = 33, True
        event.client.display, event.client.window = self.display, window
        event.client.message_type, event.client.format = self.active_atom, 32
        event.client.data[0] = 2  # Pager-style explicit activation request.
        event.client.data[2] = self.active()
        if not self.x.XSendEvent(self.display, self.root, False, (1 << 20) | (1 << 19), ctypes.byref(event)):
            raise RuntimeError("window manager rejected activation request")
        self.x.XFlush(self.display)

    def close(self):
        self.x.XCloseDisplay(self.display)


@contextmanager
def visible_capture(desktop, window, timeout=3.0, redraw=.5):
    previous = desktop.active()
    try:
        desktop.activate(window)
        deadline = time.monotonic() + timeout
        while desktop.active() != window:
            if time.monotonic() >= deadline:
                raise RuntimeError("RViz did not become the active window; refusing an occluded capture")
            time.sleep(.05)
        time.sleep(redraw)
        if desktop.active() != window:
            raise RuntimeError("active window changed before RViz capture")
        yield
        if desktop.active() != window:
            raise RuntimeError("active window changed during RViz capture")
    finally:
        # Do not steal focus back if the user switched to a different window.
        if previous and previous != window and desktop.active() == window:
            desktop.activate(previous)


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
    match, _tree = find_window(args.title)
    window_id = match.group(1)
    width, height = int(match.group(3)), int(match.group(4))
    subtree = subprocess.check_output(["xwininfo", "-id", window_id, "-tree"], text=True)
    viewport_width = render_viewport_width(subtree)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    desktop = XDesktop()
    try:
        with visible_capture(desktop, int(window_id, 16)):
            with tempfile.NamedTemporaryFile(suffix=".xwd") as capture:
                subprocess.run(["xwd", "-silent", "-id", window_id, "-out", capture.name], check=True)
                image = read_xwd(Path(capture.name))
    finally:
        desktop.close()
    image.save(args.output)
    array = np.asarray(image, dtype=np.uint8)
    # A real frame should contain more than one almost-uniform panel colour.
    spread = float(array.reshape(-1, 3).std(axis=0).mean())
    report = {
        "window_id": window_id,
        "window_title": match.group(2),
        "active_window_verified": True,
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
