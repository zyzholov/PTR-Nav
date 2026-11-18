#!/usr/bin/env python3
"""Frozen measured-map qualification of terrain/control horizon compatibility.

No truth, command publication or physical navigation claim. Refit only the
recorded conditioning scan; radius changes the model tile, never its distance,
obstacle, age or physical risk limits. CPU/GPU parity is checked separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from shared_certificate_qualification import frozen_prediction
from fsgp_bgk_se2.shared_terrain import shared_geometric_probability


def probabilities(prediction, physical, collision):
    start = time.perf_counter()
    result = np.full((physical.yaw_bins, *prediction.mean.shape), np.nan, np.float32)
    for k, (r, c, values) in enumerate(shared_geometric_probability(prediction, physical, collision)):
        result[k, r, c] = values
    return result, (time.perf_counter()-start)*1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--radius', type=float, default=5.)
    p.add_argument('--basis-spacing', type=float, default=None)
    a = p.parse_args()
    if not math.isfinite(a.radius) or not 1 <= a.radius <= 5 or a.output.exists():
        p.error('new output and radius 1..5m required')
    if a.basis_spacing is not None and (not math.isfinite(a.basis_spacing) or not .5 <= a.basis_spacing <= 2.):
        p.error('basis spacing .5..2m required')
    torch.set_num_threads(1)
    records = {}; fields = {}; physical = None
    for name, device, radius, verify in (
            ('archived_gpu', 'cuda', None, True),
            ('expanded_gpu', 'cuda', a.radius, False),
            ('expanded_cpu', 'cpu', a.radius, False)):
        prediction, physical, collision, provenance = frozen_prediction(a.snapshot, device,
            replay_initial=True, verify_posterior=verify, posterior_radius_override=radius,
            basis_spacing_override=None if verify else a.basis_spacing)
        values, elapsed = probabilities(prediction, physical, collision)
        x, y, _, yaw = provenance['query_pose']; left, bottom = provenance['map_min']
        probes = []
        for distance in (0., 1., 2., 2.75, 3., 3.6, 4.2, 4.8):
            row = int((y+distance*math.sin(yaw)-bottom)/physical.resolution)
            col = int((x+distance*math.cos(yaw)-left)/physical.resolution)
            k = int((yaw % (2*math.pi))/(2*math.pi/physical.yaw_bins))
            upper = float(values[k, row, col])
            probes.append(dict(ahead_m=distance, conditional_terrain_upper=upper if math.isfinite(upper) else None,
                state_within_5_percent=bool(math.isfinite(upper) and upper <= .05)))
        records[name] = dict(provenance=provenance, training_points=prediction.training_points,
            fit_wall_ms=prediction.fit_ms, probability_wall_ms=elapsed, probes=probes)
        fields[name] = (prediction.valid, values)
    cpu, gpu = fields['expanded_cpu'], fields['expanded_gpu']
    finite = np.isfinite(cpu[1]) & np.isfinite(gpu[1])
    delta = float(np.max(abs(cpu[1][finite]-gpu[1][finite]))) if finite.any() else None
    parity = dict(support_equal=bool(np.array_equal(cpu[0], gpu[0])),
        finite_probability_equal=bool(np.array_equal(np.isfinite(cpu[1]), np.isfinite(gpu[1]))),
        maximum_probability_delta=delta,
        decision_equal=bool(np.array_equal(cpu[1] <= .05, gpu[1] <= .05)))
    accepted = parity['support_equal'] and parity['finite_probability_equal'] and parity['decision_equal'] and delta is not None and delta < 5e-5
    result = dict(scope=__doc__, accepted=accepted, models=records, parity=parity,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat='Parity qualifies implementation only; expanded posterior is a different model, not proof of physical success or calibrated uncertainty.')
    a.output.mkdir(parents=True)
    (a.output/'qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(accepted=accepted, parity=parity, probes={k:v['probes'] for k,v in records.items()})))
    return int(not accepted)


if __name__ == '__main__':
    raise SystemExit(main())
