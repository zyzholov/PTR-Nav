"""Geometry shared by the Isaac sensor authoring and FAST-LIO qualification."""

import math


DEFAULT_PITCH_DEG = 25.0
LIVOX_BASE_TRANSLATE = (0.18, 0.0, 0.250)
LIDAR_FRAME_TRANSLATE = (-0.012, 0.0, 0.047)
IMU_LINK_TRANSLATE = (0.18, 0.0, 0.205)


def mount_pitch_degrees(value):
    pitch = float(value)
    if not math.isfinite(pitch) or not -90.0 <= pitch <= 90.0:
        raise ValueError("LiDAR mounting pitch must be finite and within ±90 degrees")
    return pitch


def lidar_to_imu(pitch_deg):
    """Return the row-major rotation and translation for p_imu = R p_lidar + t."""
    angle = math.radians(pitch_deg)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = LIDAR_FRAME_TRANSLATE
    bx, by, bz = LIVOX_BASE_TRANSLATE
    ix, iy, iz = IMU_LINK_TRANSLATE
    return ([c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c],
            [bx - ix + c*x + s*z, by - iy + y, bz - iz - s*x + c*z])
