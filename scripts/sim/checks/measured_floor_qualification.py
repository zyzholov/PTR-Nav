#!/usr/bin/env python3
"""One archived-frame qualification of measured-floor/column separation.

Reproduce the old posterior first, then change ONLY ground-label selection.
No data or reference route is published. Pure terrain bounds exclude the
separate deterministic body collision test and are NOT navigation outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from shared_certificate_qualification import frozen_prediction
from fsgp_bgk_se2.shared_terrain import shared_geometric_probability


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    hashes={n:hashlib.sha256((a.snapshot/n).read_bytes()).hexdigest()
        for n in ('map.cdr','completion_cloud.cdr','odom.cdr')}
    old,params,collision,baseline=frozen_prediction(a.snapshot,a.device,replay_initial=True)
    new,_,after_collision,changed=frozen_prediction(a.snapshot,a.device,replay_initial=True,
        measured_floor_labels_override=True,verify_posterior=False)
    if a.device=='cuda' and new.factor.device.type!='cuda':raise ValueError('requested GPU model is not on CUDA')
    np.testing.assert_array_equal(collision,after_collision)
    def bounds(q):
        out=np.full((params.yaw_bins,*collision.shape),np.nan,np.float32)
        for k,(r,c,u) in enumerate(shared_geometric_probability(q,params,np.zeros_like(collision))):out[k,r,c]=u
        return out
    before,after=bounds(old),bounds(new)
    record=dict(scope=__doc__,device=str(new.factor.device),source_sha256=hashes,baseline=baseline,
        changed=changed,collision_unchanged=True,
        posterior_height_change_max_m=float(np.nanmax(np.abs(new.mean-old.mean))),
        eligible_terrain_poses_before=int(np.isfinite(before).sum()),
        eligible_terrain_poses_after=int(np.isfinite(after).sum()),
        terrain_bound_le_005_before=int((before<=.05).sum()),
        terrain_bound_le_005_after=int((after<=.05).sum()),
        interpretation='Terrain constraints only, before exact full-body occupancy. New labels need not reduce every risk value; actual measurements can also reveal risk.')
    if a.device=='cuda':
        cpu,_,_,_=frozen_prediction(a.snapshot,'cpu',replay_initial=True,
            measured_floor_labels_override=True,verify_posterior=False)
        np.testing.assert_array_equal(cpu.valid,new.valid)
        error=float(np.nanmax(abs(cpu.mean-new.mean)))
        variance_error=float(np.nanmax(abs(cpu.variance-new.variance)))
        if error>3e-5 or variance_error>2e-6:raise ValueError('CPU/GPU posterior mismatch')
        record['cpu_gpu_mean_max_difference_m']=error
        record['cpu_gpu_variance_max_difference_m2']=variance_error
    np.savez_compressed(a.output/'terrain_bounds.npz',before=before,after=after,
        old_height=old.mean,new_height=new.mean,old_variance=old.variance,new_variance=new.variance,
        old_support=old.valid,new_support=new.valid,collision=collision)
    if hashes!={n:hashlib.sha256((a.snapshot/n).read_bytes()).hexdigest() for n in hashes}:
        raise ValueError('archived input changed')
    (a.output/'qualification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('baseline','changed')},ensure_ascii=False))


if __name__=='__main__':main()
