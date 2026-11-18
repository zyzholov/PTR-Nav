"""Bounded, deferred clearing evidence. Independent observer ONLY.

Actual actor poses define assessment times, never navigation inputs. During
motion we retain message references; serialization/disk I/O waits until stop.
"""
import hashlib
import json
import math
from pathlib import Path


class ClearingEvidence:
    offsets=(0.,1.,2.,4.)
    topics={'laser_map':'/Laser_map','registered':'/cloud_registered',
            'risk':'/car_0_fsgp_bgk/fused_map'}

    def __init__(self,enabled):
        self.enabled=enabled;self.entered=False;self.vacated=None
        self.samples={};self.latest={};self.out_of_order=0

    def actor(self,stamp,x,y):
        if not self.enabled or not all(math.isfinite(a) for a in (stamp,x,y)):return
        inside=abs(x-4.3)<=1.4 and abs(y)<=1.4
        if inside:self.entered=True
        elif self.entered and self.vacated is None:self.vacated=stamp

    def retain(self,kind,msg,stamp,receipt):
        if not self.enabled:return
        if kind not in self.topics:raise ValueError('unknown clearing evidence topic')
        if not all(math.isfinite(a) for a in (stamp,receipt)) or stamp<=0:raise ValueError('invalid source time')
        if kind in self.latest and stamp<self.latest[kind][1]:
            self.out_of_order+=1;return
        sample=(msg,stamp,receipt);self.latest[kind]=sample
        if self.vacated is None:return
        elapsed=stamp-self.vacated
        for offset in self.offsets:
            key=(kind,offset)
            if key in self.samples or elapsed<offset:continue
            # A late first frame may not impersonate four timely snapshots.
            self.samples[key]=sample if elapsed-offset<=.75 else None

    def archive(self,directory,serialize):
        if not self.enabled:return None
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        rows=[]
        for (kind,offset),sample in self.samples.items():
            if sample is None:
                rows.append(dict(topic=self.topics[kind],requested_age_s=offset,status='missed_source_time_window'))
                continue
            rows.append(self._write(directory,kind,f'{kind}_{offset:g}s',offset,sample,serialize))
        for kind,sample in self.latest.items():
            rows.append(self._write(directory,kind,f'{kind}_terminal',None,sample,serialize))
        data=dict(scope='assessment-only; no data returned to navigation; no clearing acceptance inferred from capture',
            region_world_xy=dict(center=[4.3,0.],half_extent=[.8,.8]),
            departure_rule='observed cylinder has entered then left the region enlarged by radius and raster reserve',
            vacated_stamp_s=self.vacated,out_of_order_samples=self.out_of_order,snapshots=rows)
        path=directory/'manifest.json';path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        return str(path)

    def _write(self,directory,kind,label,offset,sample,serialize):
        msg,stamp,receipt=sample;payload=serialize(msg);path=directory/(label+'.cdr');path.write_bytes(payload)
        return dict(topic=self.topics[kind],requested_age_s=offset,status='captured',
            source_stamp_s=stamp,receipt_stamp_s=receipt,source_age_s=receipt-stamp,
            actual_age_after_departure_s=stamp-self.vacated if self.vacated is not None else None,
            path=path.name,sha256=hashlib.sha256(payload).hexdigest())
