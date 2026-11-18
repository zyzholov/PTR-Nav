#!/usr/bin/env python3
"""Paired source-frozen motion replay; no truth enters either detector."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from score_motion_replay import score

ROOT=Path(__file__).resolve().parents[3]
HEADER='src/planner/terrain_nav/include/terrain_nav/motion_guard.hpp'


def sha(payload):return hashlib.sha256(payload).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--static',type=Path,nargs='+',required=True)
    p.add_argument('--crossing',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(args.baseline_snapshot) as archive:old=archive.read(HEADER)
    current=(ROOT/HEADER).read_bytes();cpp=ROOT/'src/planner/terrain_nav/test/motion_guard_replay.cpp'
    result=dict(scope='three frozen recordings, not independent navigation missions or probability calibration',
        qualification_script_sha256=sha(Path(__file__).read_bytes()),
        baseline_snapshot_sha256=sha(args.baseline_snapshot.read_bytes()),harness_sha256=sha(cpp.read_bytes()),
        source_sha256=dict(baseline=sha(old),candidate=sha(current)),cases=[])
    for name,content in (('baseline',old),('candidate',current)):
        include=args.output/name/'include';target=include/'terrain_nav/motion_guard.hpp'
        target.parent.mkdir(parents=True);target.write_bytes(content)
        binary=args.output/name/'replay'
        subprocess.run(['/usr/bin/g++','-std=c++17','-O2','-I',str(include),
            '-DMOTION_GUARD_SHA="'+sha(content)+'"',str(cpp),'-o',str(binary)],check=True)
    for i,path in enumerate([*args.static,args.crossing]):
        raw=path.read_bytes();capture=json.loads(raw);g=capture['frames'][0]['geometry']
        factor=min(4,max(1,math.ceil(.2/g[4]-1e-6)))
        while g[0]%factor or g[1]%factor:factor-=1
        case=dict(path=str(path.resolve()),input_sha256=sha(raw),static=i<len(args.static),
                  aggregation_factor=factor,methods={})
        for name in ('baseline','candidate'):
            output=args.output/name/f'case_{i}.json'
            subprocess.run([str(args.output/name/'replay'),str(path),str(output),str(factor)],check=True)
            replay=json.loads(output.read_text());frames=replay['frames']
            metrics=dict(frames=len(frames),motion_frames=sum(r['moving_components']>0 for r in frames),
                provisional_frames=sum(bool(r['brake_only_tracks']) for r in frames),
                compute_ms_p95=float(np.percentile([r['compute_ms'] for r in frames],95)))
            if not case['static']:
                metrics['crossing']=score(capture,replay)['summary']
            case['methods'][name]=metrics
        if sha(path.read_bytes())!=case['input_sha256']:raise ValueError('input changed during qualification')
        result['cases'].append(case)
    result['source_unchanged']=current==(ROOT/HEADER).read_bytes() and result['harness_sha256']==sha(cpp.read_bytes())
    result['static_no_confirmed_motion']=all(c['methods']['candidate']['motion_frames']==0 for c in result['cases'] if c['static'])
    c=result['cases'][-1]['methods']
    result['crossing_not_degraded']=c['candidate']['crossing']['lane_warning_fraction_during_exposure']>=c['baseline']['crossing']['lane_warning_fraction_during_exposure']
    result['accepted']=result['source_unchanged'] and result['static_no_confirmed_motion'] and result['crossing_not_degraded']
    (args.output/'qualification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    return int(not result['accepted'])


if __name__=='__main__':raise SystemExit(main())
