"""Advance the recorded MID360 ray directions; never manufacture hit points.

The bundled Gazebo CSV has 800,000 consecutive directions (four seconds at
200 kshots/s). Its first column is a sample ordinal despite the ``Time/s``
heading. RTX performs the actual intersections. This is a finite, repeating
scan-table approximation, not a model of the proprietary optical mechanism or
of intra-frame motion distortion.
"""
from pathlib import Path
import hashlib

import numpy as np


SHOTS_PER_SECOND = 200_000
SCAN_HZ = 10
SHOTS_PER_FRAME = SHOTS_PER_SECOND // SCAN_HZ
DEFAULT_CSV = Path(__file__).resolve().parents[2] / 'simulator/livox_mid360/scan_mode/mid360.csv'


class Mid360ScanTable:
    def __init__(self, path=DEFAULT_CSV):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        rows = np.loadtxt(self.path, delimiter=',', skiprows=1, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 3 or not np.isfinite(rows).all():
            raise ValueError('MID360 scan CSV must contain finite sample, azimuth, zenith columns')
        if len(rows) % SHOTS_PER_FRAME or not np.all(np.diff(rows[:, 0]) == 1):
            raise ValueError('MID360 scan table must contain contiguous complete 20k-shot frames')
        self.azimuth = np.ascontiguousarray(rows[:, 1].reshape(-1, SHOTS_PER_FRAME), dtype=np.float32)
        self.elevation = np.ascontiguousarray((90. - rows[:, 2]).reshape(-1, SHOTS_PER_FRAME), dtype=np.float32)
        self.frame_count = len(self.azimuth)
        # Uniform emission slots, all strictly inside the 100 ms frame.
        self.fire_time_ns = np.arange(SHOTS_PER_FRAME, dtype=np.uint32) * (1_000_000_000 // SHOTS_PER_SECOND)

    def phase_at(self, simulation_time):
        if not np.isfinite(simulation_time) or simulation_time < 0:
            raise ValueError('scan time must be finite and nonnegative')
        return int(np.floor(simulation_time * SCAN_HZ + 1e-7)) % self.frame_count


class RtxMid360Scan:
    """One OmniLidar emitter state, changed before each actual RTX render.

    Isaac 5.1 supports changing emitter angles with a fixed array length. The
    deprecated Camera/JSON sensor cannot expose these runtime attributes.
    Cache Vt arrays once, and update only two USD attributes per 100 ms slot.
    """
    def __init__(self, sensor, table):
        from pxr import Vt
        self.table = table
        self.azimuth = [Vt.FloatArray.FromNumpy(x) for x in table.azimuth]
        self.elevation = [Vt.FloatArray.FromNumpy(x) for x in table.elevation]
        prefix = 'omni:sensor:Core:'
        values = {
            'scanType': 'SOLID_STATE', 'scanRateBaseHz': SCAN_HZ,
            'reportRateBaseHz': SCAN_HZ, 'numberOfEmitters': SHOTS_PER_FRAME,
            # RTX detector slots, NOT physical Livox laser/ring IDs. Using
            # unique slots is required by the solid-state RTX output buffer.
            'numberOfChannels': SHOTS_PER_FRAME, 'numLines': 1, 'numRaysPerLine': [SHOTS_PER_FRAME],
            'stateResolutionStep': 1, 'maxReturns': 1,
            'nearRangeM': .1, 'farRangeM': 70., 'rangeCount': 1,
            'rangesMinM': [.1], 'rangesMaxM': [70.],
            'rangeAccuracyM': .02, 'rangeResolutionM': .02,
            'rayType': 'IDEALIZED', 'intensityProcessing': 'NORMALIZATION',
            'minReflectance': .1, 'minReflectionRangeM': 40.,
            'avgPowerW': .0065, 'waveLengthNm': 905., 'pulseTimeNs': 6,
            'azimuthErrorMean': 0., 'azimuthErrorStd': .01,
            'elevationErrorMean': 0., 'elevationErrorStd': .01,
            # IsaacCreateRTXLidarScanBuffer consumes spherical GMO elements
            # and converts them to Cartesian PointCloud2. Cartesian GMO is
            # interpreted as spherical by the Isaac 5.1 bridge and warps even
            # a horizontal floor. The public ROS output remains XYZ metres.
            'elementsCoordsType': 'SPHERICAL', 'outputFrameOfReference': 'SENSOR',
            'outputMotionCompensationState': 'NONCOMPENSATED',
            'skipDroppingInvalidPoints': False,
            'emitterState:s001:fireTimeNs': Vt.UIntArray.FromNumpy(table.fire_time_ns),
            'emitterState:s001:channelId': Vt.UIntArray.FromNumpy(
                np.arange(1, SHOTS_PER_FRAME + 1, dtype=np.uint32)),
            'emitterState:s001:rangeId': Vt.UIntArray.FromNumpy(np.zeros(SHOTS_PER_FRAME, dtype=np.uint32)),
            'emitterState:s001:bank': Vt.UIntArray.FromNumpy(np.zeros(SHOTS_PER_FRAME, dtype=np.uint32)),
        }
        for name, value in values.items():
            attr = sensor.GetAttribute(prefix + name)
            if not attr or not attr.Set(value):
                raise RuntimeError(f'Cannot configure RTX MID360 attribute: {name}')
        sensor.GetAttribute('omni:sensor:tickRate').Set(float(SCAN_HZ))
        self.az_attr = sensor.GetAttribute(prefix + 'emitterState:s001:azimuthDeg')
        self.el_attr = sensor.GetAttribute(prefix + 'emitterState:s001:elevationDeg')
        self.phase = None
        self.update(0.)

    def update(self, simulation_time):
        phase = self.table.phase_at(simulation_time)
        if phase == self.phase:
            return False
        if not self.az_attr.Set(self.azimuth[phase]) or not self.el_attr.Set(self.elevation[phase]):
            raise RuntimeError('Cannot advance MID360 RTX firing pattern')
        self.phase = phase
        return True
