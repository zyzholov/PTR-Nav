#!/usr/bin/env python3
"""Five controlled GP-to-production-planner checks, not physical navigation.

No hidden heights condition the posterior. The native harness separately checks
paths against analytic terrain. These fixtures do not establish calibration or
an Isaac arrival rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','prediction_backend:=feature_sgp',
        '-p','gp_async:=false','-p',f'gp_device:={args.device}','-p','gpu_backend:='+
        ('cupy' if args.device=='cuda' else 'cpu'),'-p','gp_complete_supported:=true',
        '-p','probability_use_gp_completion:=true','-p','probability_model:=shared_gp',
        '-p','probability_gp_max_distance_m:=0.75'])
    node=FsgpBgkSe2Node();cases=[];measurements={}
    try:
        x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
        for name in ('measured_plane','enclosed_blind_spot','missing_strip','one_sided_unknown','half_metre_step'):
            observed=np.ones(x.shape,bool);z=np.zeros(x.shape)
            if name=='enclosed_blind_spot':observed=np.hypot(x,y)>.6
            if name=='missing_strip':observed=np.abs(x)>.2
            if name=='one_sided_unknown':observed=x>.6
            if name=='half_metre_step':z=np.where(x>=0,.5,0.)
            msg=SE2Grid();msg.info.header.frame_id='map';msg.info.header.stamp.sec=10
            msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=6.
            msg.info.length_yaw=2*np.pi;msg.info.yaw_resolution=2*np.pi/36
            for layer,value in (('elevation',np.where(observed,z,np.nan)),('var',np.full(x.shape,.0001)),
                ('valid',observed.astype(float)),('ground_observed',observed.astype(float)),
                ('risk',np.where(observed,0.,.65))):
                msg.layers.append(layer);msg.has_so2.append(False)
                msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
            node.latest_cloud=create_cloud_xyz32(Header(frame_id='map',stamp=msg.info.header.stamp),
                np.c_[x[observed],y[observed],z[observed]])
            with patch.object(node,'_prediction_for_map',return_value=None):
                begin=time.perf_counter();node._process_map(msg)
                cold_ms=(time.perf_counter()-begin)*1000
                begin=time.perf_counter();_,out=node._process_map(msg)
                hot_ms=(time.perf_counter()-begin)*1000
            payload=serialize_message(out);(args.output/(name+'.cdr')).write_bytes(payload)
            measurements[name]=dict(cold_ms=cold_ms,hot_ms=hot_ms,
                stage_ms=dict(node._process_stage_ms),cdr_sha256=hashlib.sha256(payload).hexdigest())
            cases.append(dict(id=name,scenario=name,layout_id=0,seed=71,
                expected_stop=name in ('one_sided_unknown','half_metre_step'),
                start=[-1.5,0.,0.],goal=[1.5,0.,0.],map_file=name+'.cdr',obstacles=[],
                terrain=dict(kind='step' if name=='half_metre_step' else 'flat',
                    scale=.5 if name=='half_metre_step' else 0.,edge=0.)))
        plan=dict(settings=dict(timing_repeats=1,qualification_only=True),parameters=dict(
            use_sim_time=True,chance_constrained=True,chance_allow_gp_completion=True,
            chance_allow_joint_model=True,live_obstacle_enable=False,
            chance_state_limit=.05,chance_path_budget=.10),cases=cases)
        (args.output/'cases.json').write_text(json.dumps(plan,indent=2)+'\n')
        (args.output/'enhancement.json').write_text(json.dumps(measurements,indent=2)+'\n')
    finally:node.destroy_node();rclpy.shutdown()
    root=Path(__file__).resolve().parents[3]
    subprocess.run([str(root/'build/terrain_nav/terrain_nav_query_benchmark'),
        str(args.output/'cases.json'),str(args.output/'planning')],check=True)
    results=json.loads((args.output/'planning/results.json').read_text())['runs']
    failures=[]
    for row in results:
        expected=row['id'] not in ('one_sided_unknown','half_metre_step')
        if bool(row.get('full_goal_connector'))!=expected or (expected and not row.get('independent_path_safe')):
            failures.append(row['id'])
    summary=dict(passed=not failures,failures=failures,scope=__doc__,cases=[{k:r.get(k) for k in (
        'id','returned','full_goal_connector','independent_path_safe','planner_wall_ms','goal_residual_m')}
        for r in results],enhancement=measurements)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return int(bool(failures))


if __name__=='__main__':raise SystemExit(main())
